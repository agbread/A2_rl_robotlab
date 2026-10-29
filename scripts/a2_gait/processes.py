"""Scoped process ownership and Linux advisory locks. No shell execution."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import threading

CANCELLED = threading.Event()


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"An experiment already owns {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_environment(gpu):
    env = dict(os.environ)
    # Isaac's graphics device IDs are physical, while a CUDA visibility mask
    # renumbers CUDA IDs. Use physical IDs consistently instead of mixing them.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONUNBUFFERED="1", MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(gpu))
    # Independent workers are not torchrun ranks, even in a scheduler shell.
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(name, None)
    return env


def stop_owned(processes):
    live = [p for p in processes if p.poll() is None]
    for p in live:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60
    while any(p.poll() is None for p in live) and time.monotonic() < deadline:
        time.sleep(.2)
    for p in live:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        p.wait()


def run_jobs(jobs, deadline, monitor=None):
    """Jobs contain argv, env, cwd, log; signals touch only our new groups."""
    processes, streams = [], []
    try:
        if CANCELLED.is_set():
            raise RuntimeError("Experiment cancelled")
        if time.time() >= deadline:
            raise TimeoutError("Experiment deadline has expired")
        for job in jobs:
            log = Path(job["log"])
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("x")
            streams.append(stream)
            processes.append(subprocess.Popen(job["argv"], cwd=job["cwd"], env=job["env"],
                                              stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            if CANCELLED.is_set():
                raise RuntimeError("Experiment cancelled; stopping owned workers")
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker failed; inspect logs: {[j['log'] for j in jobs]}")
            if monitor and not monitor():
                raise RuntimeError("Resource reserve violated; stopped only owned worker groups")
            if time.time() >= deadline:
                stop_owned(processes)
                raise TimeoutError("Deadline reached; owned workers were asked to checkpoint and stop")
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f"Worker failed; inspect logs: {[j['log'] for j in jobs]}")
    finally:
        stop_owned(processes)
        for stream in streams:
            stream.close()
