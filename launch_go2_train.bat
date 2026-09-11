@echo off
cd /d C:\Users\user\go2_rl_robotlab
C:\Users\user\anaconda3\envs\go2_rl_robotlab\python.exe scripts\rsl_rl\train.py --task=RobotLab-Go2-v0 --headless --num_envs 16384 --resume --load_run 2026-09-09_14-13-47 > logs\go2_train.log 2>&1
