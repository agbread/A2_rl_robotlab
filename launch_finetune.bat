@echo off
cd /d C:\Users\user\go2_rl_robotlab
C:\Users\user\anaconda3\envs\go2_rl_robotlab\python.exe scripts\rsl_rl\train.py --task=RobotLab-A2-v0 --headless --num_envs 16384 --resume --load_run 2026-08-25_18-59-58 --checkpoint model_163500.pt > logs\finetune_stairs.log 2>&1
