"""Render evaluate_mujoco.py trajectories without re-running or changing physics.

Example (EGL may require GPU device access outside the process sandbox):
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/a2_stairs/render_mujoco_evaluation.py \
      --evaluation outputs/my_run/evaluation --output outputs/my_run/videos \
      --checkpoint logs/my_run/model_1999.pt

Each video uses actual saved states at real-time playback speed, followed by an
explicit two-second TEST ENDED freeze frame. Short comparison panels hold that
end card; they do not imply that simulation continued. No policy/physics tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_mujoco import ROOT, SCENE, sha256, validate_scene

sys.path.insert(0, str(ROOT / "scripts/a2_gait"))
from video_assets import find_video_font


def scene_for_terrain(terrain):
    return {"stairs": SCENE, "flat": ROOT / "resources/a2_description/flat.xml"}[terrain]


def end_detail(result):
    if result["terrain"] == "flat":
        return ("Flat walking completed; straight-path accuracy not certified" if result["success"]
                else "Flat walking ended early; see termination status")
    if result["success"]:
        return "Ascent and standing check passed"
    return ("Ascent complete; four-foot standing check NOT passed" if result["ascent_completed"]
            else "Ascent not completed")


def frame_indices(times, fps):
    """Nearest saved state, with real-time duration rounded up to one frame."""
    requested = np.arange(math.ceil(float(times[-1]) * fps - 1e-8)) / fps
    right = np.searchsorted(times, requested).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    return np.where(abs(times[left] - requested) <= abs(times[right] - requested), left, right)


def annotate(pixels, raw, index, result, fonts, checkpoint_label, ended=False):
    frame = Image.fromarray(pixels)
    draw = ImageDraw.Draw(frame)
    width, height = frame.size
    title, small = fonts
    draw.rectangle((0, 0, width, 88), fill=(16, 23, 34))
    draw.text((20, 10), f"A2 | MuJoCo Sim2Sim | vx = {result['speed']:.1f} m/s", font=title, fill="white")
    cmd = raw["command"][index]
    terrain_label = "flat ground" if result["terrain"] == "flat" else "12 stairs"
    draw.text((20, 52), f"Body-frame command: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]  |  single robot  |  {terrain_label}", font=small, fill=(180, 212, 240))
    draw.rectangle((0, height - 100, width, height), fill=(16, 23, 34))
    velocity = raw["velocity"][index]
    pos = raw["base_pos"][index]
    yaw = raw["rpy_deg"][index, 2]
    draw.text((20, height - 88), f"t = {raw['time'][index]:5.2f} s   actual vx = {velocity[0]:+.2f}   vy = {velocity[1]:+.2f} m/s   yaw = {yaw:+.1f} deg", font=small, fill="white")
    draw.text((20, height - 56), f"World position: x = {pos[0]:.2f}   y = {pos[1]:+.2f}   z = {pos[2]:.2f} m   |   checkpoint: {checkpoint_label}", font=small, fill=(180, 212, 240))
    if result["terrain"] == "flat":
        draw.text((20, height - 28), "Flat ground | constant command | no resets or heading correction", font=small, fill=(180, 212, 240))
    elif result["ascent_completed_time_s"] is not None and raw["time"][index] >= result["ascent_completed_time_s"]:
        draw.text((20, height - 28), "Ascent complete | command zero | checking four-foot standing", font=small, fill=(101, 227, 156))
    if ended:
        color = (101, 227, 156) if result["success"] else (255, 165, 120)
        draw.rectangle((width // 2 - 490, height // 2 - 75, width // 2 + 490, height // 2 + 75), fill=(16, 23, 34))
        label = f"TEST ENDED: {result['status'].upper().replace('_', ' ')}"
        draw.text((width // 2, height // 2 - 45), label, font=title, fill=color, anchor="mm")
        draw.text((width // 2, height // 2), end_detail(result), font=small, fill=color, anchor="mm")
        draw.text((width // 2, height // 2 + 40), "Final state held for viewing; simulation stopped", font=small, fill="white", anchor="mm")
    return np.asarray(frame)


def render_case(result, evaluation, output, model, renderer, camera, fonts, args):
    name = result["name"]
    with np.load(evaluation / f"{name}.npz", allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}
    if not all(np.isfinite(value).all() for value in raw.values()):
        raise ValueError(f"Nonfinite trajectory: {name}")
    np.testing.assert_allclose(raw["time"][-1], result["duration_s"], atol=1e-8)
    data = mujoco.MjData(model)
    filename = output / f"a2_mujoco_vx_{result['speed']:.1f}.mp4"
    indices = frame_indices(raw["time"], args.fps)
    hold_frames = round(args.end_hold * args.fps)

    def render(index, ended=False):
        data.qpos[:] = raw["qpos"][index]
        data.qvel[:] = raw["qvel"][index]
        data.time = float(raw["time"][index])
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.qpos[:3] + np.array([0.35, 0, 0.12])
        renderer.update_scene(data, camera=camera)
        return annotate(renderer.render(), raw, index, result, fonts, args.checkpoint.stem, ended)

    writer = imageio.get_writer(str(filename), fps=args.fps, codec="libx264", quality=None,
                               macro_block_size=1, ffmpeg_params=["-crf", "20", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"])
    try:
        for frame_id, index in enumerate(indices):
            pixels = render(index)
            writer.append_data(pixels)
            if frame_id in (0, len(indices) // 2):
                Image.fromarray(pixels).save(output / f"{name}_{'start' if frame_id == 0 else 'middle'}.jpg", quality=92)
            if frame_id % (5 * args.fps) == 0:
                print(f"Rendering {name}: {frame_id}/{len(indices)} frames", flush=True)
        final_pixels = render(len(raw["time"]) - 1, ended=True)
        Image.fromarray(final_pixels).save(output / f"{name}_end.jpg", quality=92)
        for _ in range(hold_frames):
            writer.append_data(final_pixels)
    finally:
        writer.close()
    moving = (raw["time"] >= 1) & (raw["command"][:, 0] > 0)
    stats = {"terrain": result["terrain"], "speed_m_s": result["speed"], "status": result["status"],
             "ascent_completed": result["ascent_completed"], "simulation_duration_s": result["duration_s"],
             "ascent_completed_time_s": result["ascent_completed_time_s"],
             "mean_moving_vx": float(raw["velocity"][moving, 0].mean()) if moving.any() else None,
             "mean_moving_vy": float(raw["velocity"][moving, 1].mean()) if moving.any() else None,
             "final_x_m": float(raw["base_pos"][-1, 0]), "final_y_m": float(raw["base_pos"][-1, 1]),
             "final_yaw_deg": float(raw["rpy_deg"][-1, 2]), "file": filename.name,
             "max_abs_world_y_m": float(abs(raw["base_pos"][:, 1]).max()),
             "max_abs_yaw_deg": float(abs(raw["rpy_deg"][:, 2]).max()),
             "frames": len(indices) + hold_frames, "video_duration_s": (len(indices) + hold_frames) / args.fps,
             "trajectory_sha256": sha256(evaluation / f"{name}.npz")}
    print(json.dumps(stats), flush=True)
    return stats


def comparison(videos, output, fps):
    target_frames = max(video["frames"] for video in videos)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-filter_complex_threads", "2"]
    for video in videos:
        command += ["-i", str(output / video["file"])]
    filters = [f"[{i}:v]tpad=stop_mode=clone:stop={target_frames - video['frames']},setpts=PTS-STARTPTS[v{i}]" for i, video in enumerate(videos)]
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]")
    destination = output / "a2_mujoco_four_speeds_comparison.mp4"
    command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20",
                "-preset", "fast", "-pix_fmt", "yuv420p", "-threads", "4", "-r", str(fps), "-movflags", "+faststart", str(destination)]
    subprocess.run(command, check=True)
    return destination.name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--end-hold", type=float, default=2)
    args = parser.parse_args()
    if args.width != 1280 or args.height != 720 or not 1 <= args.fps <= 50 or args.end_hold < 1:
        parser.error("HUD is designed for 1280x720; fps must be 1..50 and end-hold >=1 second")
    args.evaluation, args.output, args.checkpoint = args.evaluation.resolve(), args.output.resolve(), args.checkpoint.resolve()
    manifest = json.loads((args.evaluation / "manifest.json").read_text())
    for path, digest in manifest["sources_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Evaluation source has changed: {path}")
    if sha256(manifest["policy_path"]) != manifest["policy_sha256"]:
        raise ValueError("Evaluated policy hash mismatch")
    results = json.loads((args.evaluation / "summary.json").read_text())
    terrains = {result["terrain"] for result in results}
    if len(terrains) != 1 or not terrains <= {"stairs", "flat"}:
        parser.error("Render one terrain per output directory: stairs or flat")
    terrain = next(iter(terrains))
    if len({f"{result['speed']:.1f}" for result in results}) != len(results):
        parser.error("Use one seed per speed, with unique speeds at one decimal place")
    scene = scene_for_terrain(terrain)
    checkpoint_hash = sha256(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if terrain == "stairs":
        validate_scene(model)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.8, 60, -30
    if terrain == "flat":
        camera.distance = 2.8
    font_path = find_video_font()
    fonts = (ImageFont.truetype(font_path, 29), ImageFont.truetype(font_path, 22))
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        videos = [render_case(result, args.evaluation, args.output, model, renderer, camera, fonts, args) for result in results]
    combined = comparison(videos, args.output, args.fps) if len(videos) == 4 else None
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=videos[0].keys())
        writer.writeheader()
        writer.writerows(videos)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
              "terrain": terrain, "scene": str(scene), "scene_sha256": sha256(scene),
              "evaluation": str(args.evaluation), "evaluation_manifest_sha256": sha256(args.evaluation / "manifest.json"),
              "renderer_sha256": sha256(__file__), "width": args.width, "height": args.height, "fps": args.fps,
              "camera": {"distance": camera.distance, "azimuth": camera.azimuth, "elevation": camera.elevation,
                         "lookat": "base world position + [0.35, 0, 0.12]"},
              "end_card_seconds": args.end_hold, "videos": videos, "comparison": combined,
              "method": "Real-time replay of the evaluated MuJoCo qpos/qvel trajectories, nearest saved 50 Hz sample; no dynamics re-simulation. Terminal cards are static and clearly labeled.",
              "limitations": ["One deterministic start and actuator-delay seed per speed, not 256 parallel environments or a success-rate benchmark.",
                              "Nominal MuJoCo deployment parameters; not identical physics/domain randomization to IsaacLab.",
                              ("Flat rollout completion is survival for the prescribed duration, not a straight-path accuracy pass. Command stays constant; no resets or heading correction."
                               if terrain == "flat" else "Existing evaluator commands zero after all four feet have landed on the top, then checks standing stability.")]}
    (args.output / "video_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved videos and provenance: {args.output}", flush=True)


if __name__ == "__main__":
    main()
