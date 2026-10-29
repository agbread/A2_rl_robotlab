<div align="center">
	<h1 align="center">A2 RL RobotLab</h1>
	<a href="https://robogauge.github.io/complete/">
		<img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"/>
	</a>
	<a href="https://robogauge.github.io/static/files/arxiv.pdf">
		<img src="https://img.shields.io/badge/Paper-RSS%202026-blue.svg" alt="RSS 2026 Paper"/>
	</a>
	<a href="https://arxiv.org/abs/2602.00678">
		<img src="https://img.shields.io/badge/arXiv-2602.00678-b31b1b.svg" alt="arXiv:2602.00678"/>
	</a>
	<a href="https://github.com/wty-yy/go2_rl_gym">
		<img src="https://img.shields.io/badge/IsaacGym-go2__rl__gym-orange.svg" alt="IsaacGym go2_rl_gym"/>
	</a>
</div>

## Overview

A2_rl_robotlab adapts IsaacLab / RobotLab locomotion training and MuJoCo Sim2Sim deployment for the Unitree A2. It is a fork of [wertyuilife2/go2_rl_robotlab](https://github.com/wertyuilife2/go2_rl_robotlab), retaining the upstream MoE-CTS implementation and Go2 support.

A2 additions include the robot task and deployment assets, exact 15 cm rise / 20 cm tread stairs, a 6-to-12-step curriculum, optional gait symmetry training, and reproducible policy comparisons. See [A2 stair fine-tuning](#a2-12-step-stair-fine-tuning) for training and evaluation commands.

The Go2 demonstrations and benchmark scores below are upstream reference results. The original paper and project are credited through the [upstream project page](https://robogauge.github.io/complete/). Original license and copyright notices are retained; see [LICENSE](LICENSE) and the [RSL-RL license](source/rsl_rl/LICENSE).

## Clone

```bash
git clone https://github.com/agbread/A2_rl_robotlab.git
cd A2_rl_robotlab
git remote add upstream https://github.com/wertyuilife2/go2_rl_robotlab.git
```

The installation and training commands use the `a2_rl_robotlab` Conda environment.

## Two-GPU gait experiment: prepare here, train on another Ubuntu PC

This workflow compares **warm-start correction** with **training from scratch**.
It uses two independent single-GPU CTS jobs, not distributed PPO. Existing tasks,
models, and deployment defaults are unchanged. The preparation machine runs only
CPU checks; **Isaac Sim startup, GPU capacity, training convergence and generated
gait quality must be validated on the destination PC.**

### Download and install on the training PC

```bash
git clone --branch a2-gait-training-prep https://github.com/agbread/A2_rl_robotlab.git
cd A2_rl_robotlab
conda create -n a2_rl_robotlab python=3.11
conda activate a2_rl_robotlab
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaaclab[isaacsim,all]==2.3.2.post1' --extra-index-url https://pypi.nvidia.com -c scripts/a2_gait/constraints.txt
python -m pip install -e source/robot_lab -e source/rsl_rl -c scripts/a2_gait/constraints.txt
python -m pip install tensordict==0.14.3 mujoco==3.17.0 numpy==1.26.0 h5py==3.16.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 pytest==9.1.1 psutil pygame scipy pillow pyyaml -c scripts/a2_gait/constraints.txt
# Install these Ubuntu packages if not already available (video encoder/font).
sudo apt-get install ffmpeg fonts-dejavu-core
```

The constraints record the existing preparation environment; a fresh installation
has not been exercised here. A compatible NVIDIA driver, Vulkan/EGL, enough disk
space, and Ubuntu's DejaVu Sans font are required. Accept NVIDIA's licenses on the
destination machine as required by its installer. Do not replace a working driver
automatically. No credentials or machine-local Conda paths are needed by the code.

The 26 MB warm-start checkpoint and provenance snapshots are included in
`deploy/pre_train/a2/gait_base_model1999/`; no separate file transfer or Git LFS is
needed. Its byte-for-byte SHA-256 is
`e929e9df83c945dc24df9888d7059a2b1c4f7293fd50d3dec7f250b2fa007a2d`.
It contains the learned teacher, student, shared actor and critic. The scratch
branch does **not** load these weights. The bundled YAML files are historical
snapshots, not the new launch recipe.

### Safe preview, then explicit remote GPU execution

```bash
# Safe on the preparation PC: no CUDA, Isaac Sim, or child jobs.
python scripts/a2_gait/manage.py preflight --static
python scripts/a2_gait/manage.py train --branch both --run-dir outputs/gait_week1 --dry-run

# Run the following ONLY on the destination PC, after reserving both GPUs.
# Use physical nvidia-smi GPU IDs in an unmasked local session.
# Do not unset a scheduler's allocation mask to access unallocated GPUs.
python scripts/a2_gait/manage.py preflight --gpu --gpus 0 1 --run-dir outputs/gait_week1
python scripts/a2_gait/manage.py benchmark --gpus 0 1 --run-dir outputs/gait_week1
python scripts/a2_gait/manage.py train --branch both --gpus 0 1 --run-dir outputs/gait_week1

# Resume this same recipe, environment count, devices and original deadlines.
python scripts/a2_gait/manage.py resume --gpus 0 1 --run-dir outputs/gait_week1
```

`preflight --gpu` runs only a small simulation probe, not optimizer updates.
`benchmark` explicitly performs disposable profiling updates on both GPUs
simultaneously: 20 warmup + 50 timed updates at 1,024/2,048/4,096 environments per
GPU. It chooses the highest combined throughput satisfying **8 GiB free per GPU
and 20 GiB free system RAM throughout the measurement**. An unsafe smallest
configuration blocks training. Profiling checkpoints are never used to initialize
either experiment. No other user's processes are stopped.

CUDA, PhysX and the renderer use the same physical GPU ID, with Kit multi-GPU
rendering disabled. The manager rejects a pre-existing `CUDA_VISIBLE_DEVICES`
mask instead of silently confusing physical and remapped IDs. Each job starts
with four CPU threads. The `--branch warm` and `--branch scratch` modes are also
available; use the same branch choice for preflight, benchmark, train and resume.

The first remote GPU preflight creates persistent deadlines: **144 hours for
preparation/training/intermediate evaluation and 168 hours overall**. Pauses and
restarts do not reset this budget. Use a fresh run directory for a deliberately
new experiment. Each directory is locked against duplicate manager instances;
interrupts affect only child process groups owned by that manager. Atomic
checkpoints can be recovered after an interrupted chunk. Changing source or
protocol during an experiment requires a new experiment, not an implicit resume.

### Recipes and progression

| Setting | Warm branch / GPU 0 | Scratch branch / GPU 1 |
|---|---|---|
| Initialization | Bundled model_1999, fresh optimizers | All networks initialized randomly |
| PPO / student LR | `1e-5` / `1e-4`, fixed | Flat `1e-3` adaptive / `1e-3`; stairs `1e-4` / `1e-4`; mixed `1e-5` / `1e-4` |
| Teacher/student environments | 75% / 25% | 75% / 25% |
| Update ceiling | 10,000; assess initial 2,000 | 300,000 or time budget |
| Save / quick phase-gate evaluation | 100 / not needed | 500 / 500 updates before mixed phase |
| Mirror + new reward ramp | 200 updates | 1,000 updates |

The shared controller still uses 450 history observations, 45 current observations,
275 privileged teacher/critic observations and 12 actions. No deployment adapter
or physics is changed to hide gait imbalance.

New rewards apply only on **flat tiles with vx > 0.2, |vy| < 0.05 and |yaw rate| <
0.05**. Front and rear left/right completed swing differences are normalized by
0.4 s, squared and averaged, with weight -0.2. Two-second contact-duty differences
are squared and averaged, with weight -1.0. Contact uses world-vertical force >10 N;
control-step majority contact defines swing spans. Missing/partial swings are not
fabricated, a permanently airborne foot is penalized through duty, and reset or
command changes clear the statistics. Mirror loss ends at 0.05. These weights are
experimental starting values, not demonstrated optimal settings.

Warm training uses 50% flat, 20% fixed 12-step target, 10% 6/8/10/12-step curriculum,
and four neighboring stair geometries at 5% each. Flat commands are 80% forward
0.5–1.2 m/s with zero lateral/yaw command and 20% legacy command replay. Stair
commands remain 0.3–0.5 m/s. Flat tile boundaries cause timeouts, not fall labels.

Scratch progression is explicit and checkpointed:

1. Flat at 0.2–0.5 m/s. Both policies must pass two consecutive 0.5 m/s evaluations:
   20-second survival >=95%, vx RMSE <=0.15 m/s.
2. Flat at 0.5–1.2 m/s. Both policies must pass at 0.5/0.7/0.9/1.2: the same
   survival/tracking thresholds, front swing ratio <=1.5 and duty gap <=15 pp,
   for two consecutive evaluations.
3. Keep 50% flat while stairs progress through 2 steps/5 cm, 4/10 cm, 6/15 cm,
   8/15 cm, 10/15 cm and 12/15 cm, always 20 cm tread. Promotion requires >=90%
   success in the last 200 **student** stair episodes plus passing flat checks.
4. Use the same final mixed recipe as the warm branch.

The original scratch base-height and vertical-velocity reward ramps are referenced
to cumulative updates. Phase changes keep neural weights, reset optimizers to the
new recipe, and preserve completed ramp progress and cumulative sample/time
counters. Same-phase resume restores both optimizers and curriculum, but begins a
new physical episode and resets gait statistics. Short-run scratch underperformance
is not evidence that scratch training is impossible; the original 219k run used
16,384 environments, so compare transitions and GPU time, not iteration labels.

### Evaluation, selection and videos

Quick paired teacher/student flat evaluations run every 500 scratch updates until
the mixed phase. Full IsaacLab+MuJoCo comparisons start at update 2,000 for warm
and update 500 for scratch, and subsequently after at least six hours. Validation
and final evaluation seeds are disjoint. The original
model_1999 is the reference in both simulators.

Isaac evaluates 64 paired starts per speed (256 environments per terrain), with
fixed 15/10 ms actuator lags and no observation noise or domain randomization.
It does not restore the training environment population or tile assignments.
MuJoCo evaluates 20 actuator-delay seeds per speed, fixed pose and unchanged
physics; these are **not independent training seeds**. Both use flat and fixed
12-step stairs at vx 0.5/0.7/0.9/1.2 with vy=yaw rate=0. Flat runs last 30 s and
statistics exclude the first 2 s. Failed cases remain in the comparison; ascent
and top-platform settling are separate outcomes.

A student candidate must pass at every speed in both simulators:

- FL mean swing at least 20% shorter than reference; front longer/shorter swing
  ratio <=1.25; front duty gap <=10 pp; rear gap worsens by no more than 5 pp.
- No increase in flat falls or mean absolute final yaw; vx RMSE worsens by no
  more than 0.05 m/s. At least 90% of episodes must contain measurable front swings.
- Stair ascent and settled-success rates each decrease by no more than 5 pp.

Final evaluation tests up to the three best validation candidates per branch,
including the last checkpoint if no full result exists for it. Passing models
are ranked by front duty gap, then vx RMSE, then final yaw, independent of their
initialization method. `selection.json` retains the baseline if none pass. A
passing selection produces four flat and four stair MuJoCo videos using seed 42.
The run limit is not a convergence guarantee and videos are not produced here.

Manual evaluation/rendering also have safe `--dry-run` previews:

```bash
python scripts/a2_gait/manage.py evaluate --checkpoint /path/to/model.pt --output outputs/my_final_eval --run-dir outputs/manual_eval --split final
python scripts/a2_gait/manage.py render --checkpoint /path/to/model.pt --output outputs/my_videos --run-dir outputs/manual_video
```

An exported checkpoint's adjacent `params/agent.yaml` must accompany it. CPU-only
preparation regression checks (synthetic tensor tests and export checks, not robot
training) are:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_a2_gait.py tests/test_a2_stair_task.py source/rsl_rl/tests -q
python -m unittest discover -s scripts/a2_stairs -p 'test_*.py' -q
```

GPU/runtime checks are intentionally deferred to the other PC. Preserve its
`experiment.json`, branch state files, `progress.json`, worker logs and benchmark
report when diagnosing an issue. Model source/protocol hashes and cumulative
environment transitions are recorded with checkpoints; logs, raw trajectories,
profiling artifacts and generated videos remain ignored by Git.

---

<p align="center">
  <img src="resources/results/isaaclab_scene.png" width="70%"/>
</p>

---

## Demos

<p align="center">
  <b>Train in IsaacLab → validate with MuJoCo Sim2Sim → evaluate on the real Unitree Go2.</b>
</p>

<table>
  <tr>
    <td width="50%" align="center">
      <b>IsaacLab Play</b><br/>
      <sub>Policy rollout in the IsaacLab environment.</sub><br/><br/>
      <img src="https://raw.githubusercontent.com/wertyuilife2/picture-bed/main/go2_rl_robotlab/demo-isaaclab.gif" width="80%"/>
    </td>
    <td width="50%" align="center">
      <b>MuJoCo Sim2Sim</b><br/>
      <sub>Exported policy running in MuJoCo before real-world deployment.</sub><br/><br/>
      <img src="https://raw.githubusercontent.com/wertyuilife2/picture-bed/main/go2_rl_robotlab/demo-sim2sim.gif" width="80%"/>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>Real Robot</b><br/>
      <sub>Robust walking front on stairs.</sub><br/><br/>
      <img src="https://raw.githubusercontent.com/wertyuilife2/picture-bed/main/go2_rl_robotlab/demo-real-walk-front.gif" width="100%"/>
    </td>
    <td width="50%" align="center">
      <b>Real Robot</b><br/>
      <sub>Robust walking sideways on stairs.</sub><br/><br/>
      <img src="https://raw.githubusercontent.com/wertyuilife2/picture-bed/main/go2_rl_robotlab/demo-real-walk-side.gif" width="100%"/>
    </td>
  </tr>
</table>

---

## RoboGauge Benchmark

<p align="center">
  <b><a href="https://github.com/wty-yy/RoboGauge">RoboGauge</a> score comparison between go2_rl_robotlab and the original go2_rl_gym.</b>
</p>

<p align="center">
  <img src="resources/results/robogauge_compare.png" width="100%"/>
</p>

### Algorithm Results (Best Checkpoint Results)

| Model | Score | Tracking  | Safety  | Quality  | Level |
| --- | --- | --- | --- | --- | --- |
| go2_moe_cts (go2_rl_robotlab) | **0.6984** | **0.7055** | **0.8159** | **0.7693** | **8.30** |
| go2_moe_cts (go2_rl_gym) | 0.6713 | 0.6669 | 0.7857 | 0.7392 | 7.85 |
| [CTS](https://arxiv.org/pdf/2405.10830) vanilla | 0.5786 | 0.5755 | 0.7066 | 0.6624 | 6.83 |
| [HIM](https://github.com/InternRobotics/HIMLoco) | 0.5379 | 0.5453 | 0.6476 | 0.6050 | 6.19 |
| [DreamWaQ](https://arxiv.org/abs/2301.10602) | 0.5054 | 0.5105 | 0.6149 | 0.5730 | 5.74 |

---

## Installation Guide

### 1. Install IsaacLab

Install IsaacLab follow [official guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/isaaclab_pip_installation.html):

```bash
conda create -n a2_rl_robotlab python=3.11
conda activate a2_rl_robotlab
pip install --upgrade pip
pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

---

### 2. Install Customized RSL-RL and RobotLab

We uses a customized version of `rsl_rl==3.3.0` and `robot_lab==2.3.0`. Install them in editable mode:

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

---

### 3. Install MuJoCo (Optional, for Sim2Sim)

To enable MuJoCo-based simulation:

```bash
pip install mujoco pygame  # tested on 3.4.0 and 3.6.0
```

---

## Training and Evaluation

Run the following commands:

```bash
# Train
python scripts/rsl_rl/train.py --task=RobotLab-Go2-v0 --headless

# Evaluate
python scripts/rsl_rl/play.py --task=RobotLab-Go2-v0
```

---

### Training with RoboGauge Evaluation

[RoboGauge](https://github.com/wty-yy/robogauge) provides an asynchronous suite for evaluating motion-control reinforcement learning policies, helping you select the best model. After installing RoboGauge by following the [installation guide](https://github.com/wty-yy/robogauge?tab=readme-ov-file#installation), start the RoboGauge server:

```bash
python robogauge/scripts/server.py --port 9973 --num-processes 32
```

In another terminal, start your training with RoboGauge evaluation enabled:

```bash
# Train
python scripts/rsl_rl/train.py --task=RobotLab-Go2-v0 --headless --robogauge --robogauge_port 9973
```

Note on Asynchronous Evaluation:

Since the evaluation runs asynchronously, the training loop receives the results from the previous evaluation in the next iteration. If evaluation is slower than training, tasks will be queued. Upon training completion, the process will wait for all remaining evaluation tasks to finish before exiting.

Performance Example: On a system with an AMD EPYC 7763 and RTX 4090, a full evaluation takes approximately 5 minutes with num_processes=63. Training and saving checkpoints every 500 steps typically takes around 30 minutes.

---

## Configuration

The training pipeline can be configured at two levels: task-level Python configuration files and runtime arguments passed to the training script.

### Task-Level Configuration

The default task settings are defined in the following files:

1. **Environment configuration**
   ```
   source/robot_lab/robot_lab/tasks/go2/env_cfg.py
   ```

2. **RL algorithm configuration**
   ```
   source/robot_lab/robot_lab/tasks/go2/rsl_rl_cfg.py
   ```

3. **Task registration**
   ```
   source/robot_lab/robot_lab/tasks/go2/__init__.py
   ```

### Runtime Overrides

In addition to the default configuration files, `train.py` and `play.py` supports several command-line arguments for runtime overrides:

```bash
python scripts/rsl_rl/train.py \
   --task=RobotLab-Go2-v0 \
   --headless \
   --experiment_name <YOUR_EXP_NAME> \
   --run_name <YOUR_RUN_NAME> \
   --num_envs <NUM_ENVS> \
   --checkpoint <PATH_TO_CHECKPOINT>
```

For more details, refer to the [robot_lab repo](https://github.com/fan-ziqi/robot_lab.git).

---

## MuJoCo Sim2Sim

Set the `policy_path` in `deploy/deploy_mujoco/configs/go2.yaml`:

```yaml
policy_path: "{ROOT_DIR}/deploy/pre_train/go2/xxx.pt" # policy.pt exported by running scripts/rsl_rl/play.py
```

Run the deployment script:

```bash
python deploy/deploy_mujoco/deploy_go2.py
```

### Controller Behavior

- **Automatic detection**: If a controller is connected, control mode is enabled automatically.
- **Fallback mode**: If no controller is detected, default commands from the config file are used.

### Controller Mapping

| Input | Function |
|------|--------|
| `LX / LY` | Forward / lateral velocity |
| `RX` | Angular velocity (steering) |

---

### Switching Simulation Scenarios

Modify `xml_path` in `deploy/deploy_mujoco/config/go2.yaml`:

```yaml
# Flat terrain
xml_path: "{ROOT_DIR}/resources/go2/flat.xml"

# Stairs
xml_path: "{ROOT_DIR}/resources/go2/stairs.xml"

# Boxes
xml_path: "{ROOT_DIR}/resources/go2/boxes.xml"

# Custom
xml_path: "{ROOT_DIR}/resources/go2/your-custom-scene.xml"
```

---

## Differences from `go2_rl_gym`

- Motor:
  - use official unitree motor model instead of simple PD controller.
- Rewards:
  - 2x tracking reward weights due to better performance.
  - different tracking reward form (fixed sigma vs. dynamic sigma).
  - lower joint_acc_l2 weight in Lab due to physics-step level implementation and sensitivity to outliers.
  - extra joint_pos_penalty_l1 reward in Lab due to better performance.
- Domain randomization: 
  - no randomized action delay, use motor-level delay instead.
  - no motor strength randomization due to implementation constraints in Lab.
- History length: 10 in Lab vs. 5 in Gym, due to better performance with longer history in Lab.
- Terrain Difficulty: use lab's continuous difficulty levels.
---

## Acknowledgements
This repository would not exist without the following open-source projects:

- [isaac_lab](https://github.com/isaac-sim/IsaacLab): Unified framework for robot learning built on NVIDIA Isaac Sim.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithms.
- [robot_lab](https://github.com/fan-ziqi/robot_lab.git): RL Extension Library for Robots, Based on IsaacLab.
- [mujoco](https://github.com/google-deepmind/mujoco.git): High-performance CPU physics simulator.

Related publications implemented in this repo:
- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)


## A2: 12-step stair fine-tuning

The A2 stair tasks use exact box meshes: 15 cm rise, 20 cm tread, 12 consecutive
upward risers (1.8 m total), a 1.5 m approach, and a 1.2 m added top landing.
The course width is 3 m. Training and MuJoCo evaluation share
`source/robot_lab/robot_lab/tasks/a2/stair_geometry.py`.

- `RobotLab-A2-Stairs-v0`: control experiment without mirror loss.
- `RobotLab-A2-Stairs-Sym-v0`: identical environment plus student-action mirror consistency.
- Terrain columns: 40% fixed target, 20% 6/8/10/12-step curriculum,
  20% neighboring dimensions (14/20, 16/20, 15/18, 15/22 cm), 20% flat.
- Stair command: 0.3–0.5 m/s, fixed for each episode; yaw and lateral commands are zero.
  Training completes after each foot has contacted the top platform and all feet are over it.
  Curriculum promotion requires at least 90 successful ascents in the latest 100 completed curriculum episodes at the current level.
  MuJoCo also measures stopping: another 0.4 s with four top contacts and tilt below 30 degrees.
  Set `env.commands.base_velocity.require_settled_completion=true` only for a dedicated stopping experiment.
- Both experiments use fixed PPO LR `1e-5` and student encoder LR `1e-4`.
  Symmetry starts at `0.05 / 200` and reaches `0.05` over 200 updates.
  This is an initial experimental weight, not a demonstrated optimum.

The original 219k A2 training checkpoint is included under [`deploy/pre_train/a2/finetune_base_219k/`](deploy/pre_train/a2/finetune_base_219k/) for fine-tuning on the A6000 PC. The examples below use this checkpoint directly. See [A6000 fine-tuning from the 219k model](#a6000-fine-tuning-from-the-219k-model). Training logs and intermediate models remain excluded from Git.

Start both experiments from the same 219k checkpoint and seed. `--warm_start`
loads model weights only, preserving fresh optimizers, curriculum and ramp clocks.
`--resume` continues a saved experiment, including those states; the flags are
mutually exclusive. Run one training job at a time on a shared GPU and choose
`--num_envs` for available memory.

```bash
conda activate a2_rl_robotlab
python scripts/rsl_rl/train.py --headless --task RobotLab-A2-Stairs-v0 \
  --num_envs 256 --max_iterations 200 --seed 42 --run_name pilot_control \
  --warm_start deploy/pre_train/a2/finetune_base_219k/model_219000.pt
python scripts/rsl_rl/train.py --headless --task RobotLab-A2-Stairs-Sym-v0 \
  --num_envs 256 --max_iterations 200 --seed 42 --run_name pilot_symmetry \
  --warm_start deploy/pre_train/a2/finetune_base_219k/model_219000.pt
```

These 200-update runs are integration pilots. They do not establish convergence
or reliable improvement in gait. Keep environment count, seed and update count
matched when extending the A/B experiment; the task default is 2,000 updates.
Compare completed ascents, forward-command velocity error, falls/course exits,
phase-aligned joint motion, contact duty and roll before choosing a checkpoint.
The mirror loss alone is not a gait or safety score.

Export a candidate and evaluate it on the same 20 MuJoCo cases (three stair
speeds × five delay seeds, plus five flat cases). Use a separate output directory
for every policy. Evaluation seeds change motor delay, not the complete range of
training domain randomization.

```bash
python scripts/a2_stairs/export_checkpoint.py --checkpoint logs/rsl_rl/EXPERIMENT/RUN/model_199.pt
python scripts/a2_stairs/evaluate_mujoco.py \
  --policy logs/rsl_rl/EXPERIMENT/RUN/exported/model_199.pt \
  --output outputs/a2_stairs_candidate --workers 2
```

Compare all three evaluation directories with `scripts/a2_stairs/compare_runs.py`.
It writes JSON plus a text table, using equal weight per evaluation run and paired
actuator-delay seeds. `scripts/a2_stairs/measure_mirror_residual.py` can additionally
compare exported policies against the same recorded observations; that open-loop
mirror error does not replace the physical rollout metrics.

To continue a stair run, use the same environment count, curriculum mode and
completion criterion. For example, `--resume --load_run RUN --checkpoint model_199.pt`
with the matching task loads optimizer, curriculum and mirror-ramp states.
`--max_iterations` specifies additional updates for that resumed invocation.
Use `--warm_start` when intentionally starting a new experiment with changed settings.

For the unchanged 219k baseline, omit `--policy`. CPU regression checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest source/rsl_rl/tests tests/test_a2_stair_task.py scripts/a2_stairs/test_evaluate_mujoco.py -q
```

### A6000 fine-tuning from the 219k model

Use [`deploy/pre_train/a2/finetune_base_219k/model_219000.pt`](deploy/pre_train/a2/finetune_base_219k/model_219000.pt)
as the starting point on the A6000 PC. This is the original A2 checkpoint at
iteration 219000, with no updates from the local stair or symmetry pilots.
The checkpoint is an unmodified copy and includes teacher/student weights and
both optimizer states. Its SHA-256 is recorded in the adjacent `metadata.json`.
The existing `deploy/pre_train/a2/a2_moe_cts_219k.pt` is the matching MuJoCo policy;
use the full `model_219000.pt` file for training.

After installing the repository on the A6000 PC, start the new stair/symmetry
fine-tuning task from the 219k weights:

```bash
conda activate a2_rl_robotlab
python scripts/rsl_rl/train.py --headless --task RobotLab-A2-Stairs-Sym-v0 \
  --num_envs 256 --max_iterations 2000 --seed 42 --run_name a6000_stairs_sym \
  --warm_start deploy/pre_train/a2/finetune_base_219k/model_219000.pt
```

`--warm_start` continues learning from the trained weights while initializing
fresh optimizers, stair curriculum, and symmetry-ramp counters for the changed
task. It does not initialize the policy from scratch. The old checkpoint does
not contain the new stair curriculum states. Use `--resume` for subsequent
continuation of the new A6000 stair run, with the same task and environment count.

The 256-environment command is a conservative starting configuration, not a
measured A6000 capacity limit. The 2,000 updates are an initial training budget,
not a convergence claim. Keep environment count, seed, and update count matched
when comparing with the control task `RobotLab-A2-Stairs-v0`.

`params/agent.yaml` and `params/env.yaml` capture the original 219k run's settings
for provenance. Local cache/export paths are made repository-relative. The new
stair task supplies its own fine-tuning settings; do not reuse the original
adaptive learning rate or resume-run references for this new experiment.

Re-export the baseline checkpoint if needed:

```bash
python scripts/a2_stairs/export_checkpoint.py \
  --checkpoint deploy/pre_train/a2/finetune_base_219k/model_219000.pt \
  --output outputs/a2_219k_reexport/policy.pt
```
