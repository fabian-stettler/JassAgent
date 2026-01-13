### Training Run

100'000 SPiele vs Rule based highest card firtst (Logs unten)
und 100'000 SPiele vs MCTS Observation Agent (8, 250)

### Logs Rule Based: 
jovyan@jupyter-fabian-stettler---af06958b:~/DL4G/JassAgent$ python -m jass.rl.train_self_play --epochs 100 --checkpoint-frequency 50
Starting training for 100 epochs, batch size 1000, MCTS device cuda
Epoch 00 | mean_reward=2.87 | games=1000 | duration=28.29s
Epoch 01 | mean_reward=-3.85 | games=1000 | duration=27.62s
Epoch 02 | mean_reward=-1.87 | games=1000 | duration=27.53s
Epoch 03 | mean_reward=-1.06 | games=1000 | duration=27.35s
Epoch 04 | mean_reward=2.98 | games=1000 | duration=27.68s
Epoch 05 | mean_reward=7.68 | games=1000 | duration=27.44s
Epoch 06 | mean_reward=9.48 | games=1000 | duration=27.36s
Epoch 07 | mean_reward=7.15 | games=1000 | duration=27.74s
Epoch 08 | mean_reward=4.58 | games=1000 | duration=27.59s
Epoch 09 | mean_reward=7.35 | games=1000 | duration=27.23s
Epoch 10 | mean_reward=6.51 | games=1000 | duration=27.27s
Epoch 11 | mean_reward=8.38 | games=1000 | duration=27.36s
Epoch 12 | mean_reward=6.56 | games=1000 | duration=27.38s
Epoch 13 | mean_reward=8.08 | games=1000 | duration=27.38s
Epoch 14 | mean_reward=10.13 | games=1000 | duration=27.29s
Epoch 15 | mean_reward=8.24 | games=1000 | duration=27.36s
Epoch 16 | mean_reward=5.92 | games=1000 | duration=27.28s
Epoch 17 | mean_reward=10.91 | games=1000 | duration=27.49s
Epoch 18 | mean_reward=10.66 | games=1000 | duration=27.66s
Epoch 19 | mean_reward=9.91 | games=1000 | duration=27.39s
Epoch 20 | mean_reward=9.94 | games=1000 | duration=27.24s
Epoch 21 | mean_reward=10.92 | games=1000 | duration=27.59s
Epoch 22 | mean_reward=9.83 | games=1000 | duration=27.83s
Epoch 23 | mean_reward=9.89 | games=1000 | duration=27.80s
Epoch 24 | mean_reward=5.07 | games=1000 | duration=27.61s
Epoch 25 | mean_reward=6.56 | games=1000 | duration=27.41s
Epoch 26 | mean_reward=11.77 | games=1000 | duration=27.44s
Epoch 27 | mean_reward=10.45 | games=1000 | duration=27.24s
Epoch 28 | mean_reward=9.06 | games=1000 | duration=27.30s
Epoch 29 | mean_reward=13.46 | games=1000 | duration=27.35s
Epoch 30 | mean_reward=15.29 | games=1000 | duration=27.41s
Epoch 31 | mean_reward=15.68 | games=1000 | duration=27.83s
Epoch 32 | mean_reward=13.63 | games=1000 | duration=27.82s
Epoch 33 | mean_reward=16.82 | games=1000 | duration=27.77s
Epoch 34 | mean_reward=17.39 | games=1000 | duration=27.65s
Epoch 35 | mean_reward=15.09 | games=1000 | duration=27.32s
Epoch 36 | mean_reward=16.14 | games=1000 | duration=27.37s
Epoch 37 | mean_reward=17.13 | games=1000 | duration=27.79s
Epoch 38 | mean_reward=17.65 | games=1000 | duration=27.78s
Epoch 39 | mean_reward=15.34 | games=1000 | duration=27.75s
Epoch 40 | mean_reward=17.86 | games=1000 | duration=27.80s
Epoch 41 | mean_reward=19.60 | games=1000 | duration=27.56s
Epoch 42 | mean_reward=18.60 | games=1000 | duration=27.21s
Epoch 43 | mean_reward=21.26 | games=1000 | duration=27.21s
Epoch 44 | mean_reward=20.11 | games=1000 | duration=27.28s
Epoch 45 | mean_reward=20.57 | games=1000 | duration=27.28s
Epoch 46 | mean_reward=17.56 | games=1000 | duration=27.24s
Epoch 47 | mean_reward=19.72 | games=1000 | duration=27.27s
Epoch 48 | mean_reward=21.29 | games=1000 | duration=27.26s
Epoch 49 | mean_reward=21.20 | games=1000 | duration=27.31s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0050.pth
Epoch 50 | mean_reward=21.62 | games=1000 | duration=27.28s
Epoch 51 | mean_reward=22.98 | games=1000 | duration=27.28s
Epoch 52 | mean_reward=22.68 | games=1000 | duration=27.28s
Epoch 53 | mean_reward=21.16 | games=1000 | duration=27.51s
Epoch 54 | mean_reward=20.21 | games=1000 | duration=27.40s
Epoch 55 | mean_reward=23.44 | games=1000 | duration=27.75s
Epoch 56 | mean_reward=22.61 | games=1000 | duration=27.73s
Epoch 57 | mean_reward=24.37 | games=1000 | duration=27.51s
Epoch 58 | mean_reward=22.65 | games=1000 | duration=27.68s
Epoch 59 | mean_reward=24.33 | games=1000 | duration=27.24s
Epoch 60 | mean_reward=24.96 | games=1000 | duration=27.23s
Epoch 61 | mean_reward=21.87 | games=1000 | duration=27.22s
Epoch 62 | mean_reward=22.81 | games=1000 | duration=27.29s
Epoch 63 | mean_reward=24.14 | games=1000 | duration=27.32s
Epoch 64 | mean_reward=23.49 | games=1000 | duration=27.37s
Epoch 65 | mean_reward=24.17 | games=1000 | duration=27.35s
Epoch 66 | mean_reward=23.78 | games=1000 | duration=27.63s
Epoch 67 | mean_reward=25.88 | games=1000 | duration=27.84s
Epoch 68 | mean_reward=25.34 | games=1000 | duration=27.74s
Epoch 69 | mean_reward=23.18 | games=1000 | duration=27.24s
Epoch 70 | mean_reward=22.13 | games=1000 | duration=27.28s
Epoch 71 | mean_reward=25.87 | games=1000 | duration=27.40s
Epoch 72 | mean_reward=23.82 | games=1000 | duration=27.37s
Epoch 73 | mean_reward=26.75 | games=1000 | duration=27.33s
Epoch 74 | mean_reward=25.94 | games=1000 | duration=27.30s
Epoch 75 | mean_reward=23.11 | games=1000 | duration=27.32s
Epoch 76 | mean_reward=26.76 | games=1000 | duration=27.34s
Epoch 77 | mean_reward=23.37 | games=1000 | duration=27.26s
Epoch 78 | mean_reward=21.97 | games=1000 | duration=27.27s
Epoch 79 | mean_reward=27.93 | games=1000 | duration=27.30s
Epoch 80 | mean_reward=28.94 | games=1000 | duration=27.45s
Epoch 81 | mean_reward=25.83 | games=1000 | duration=27.62s
Epoch 82 | mean_reward=22.02 | games=1000 | duration=27.76s
Epoch 83 | mean_reward=25.87 | games=1000 | duration=27.78s
Epoch 84 | mean_reward=23.08 | games=1000 | duration=27.51s
Epoch 85 | mean_reward=27.82 | games=1000 | duration=27.29s
Epoch 86 | mean_reward=24.35 | games=1000 | duration=27.67s
Epoch 87 | mean_reward=25.38 | games=1000 | duration=27.77s
Epoch 88 | mean_reward=23.37 | games=1000 | duration=27.77s
Epoch 89 | mean_reward=21.05 | games=1000 | duration=27.75s
Epoch 90 | mean_reward=23.84 | games=1000 | duration=27.78s
Epoch 91 | mean_reward=25.08 | games=1000 | duration=27.80s
Epoch 92 | mean_reward=23.70 | games=1000 | duration=27.79s
Epoch 93 | mean_reward=21.81 | games=1000 | duration=27.53s
Epoch 94 | mean_reward=21.16 | games=1000 | duration=27.82s
Epoch 95 | mean_reward=22.72 | games=1000 | duration=27.77s
Epoch 96 | mean_reward=19.67 | games=1000 | duration=27.75s
Epoch 97 | mean_reward=20.93 | games=1000 | duration=27.80s
Epoch 98 | mean_reward=18.27 | games=1000 | duration=27.72s
Epoch 99 | mean_reward=21.70 | games=1000 | duration=27.78s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0100.pth
Saved final checkpoint to checkpoints/jass_rl_agent_final.pth
jovyan@jupyter-fabian-stettler---af06958b:~/DL4G/JassAgent$ 


### Logs MCTS Observation Agent:
jovyan@jupyter-fabian-stettler---af06958b:~/DL4G/JassAgent$ python -m jass.rl.train_self_play --resume-from checkpoints/jass_rl_agent_final.pth --epochs 100 --checkpoint-frequency 10
Resumed weights from checkpoints/jass_rl_agent_final.pth
Starting training for 100 epochs, batch size 1000, MCTS device cuda
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
DEBUG: stats keys = ['mean_reward', 'games']
Epoch 00 | mean_reward=6.50 | games=1000 | win_rate=0.00 | current_time=2026-01-13 00:11:54 | duration=557.19s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 01 | mean_reward=8.34 | games=1000 | win_rate=0.00 | current_time=2026-01-13 00:21:12 | duration=557.52s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 02 | mean_reward=7.11 | games=1000 | win_rate=0.00 | current_time=2026-01-13 00:30:28 | duration=555.61s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 03 | mean_reward=1.01 | games=1000 | win_rate=0.00 | current_time=2026-01-13 00:39:48 | duration=559.78s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 04 | mean_reward=3.85 | games=1000 | win_rate=0.00 | current_time=2026-01-13 00:49:05 | duration=557.23s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 05 | mean_reward=7.42 | games=1000 | win_rate=0.00 | current_time=2026-01-13 00:58:19 | duration=554.29s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 06 | mean_reward=8.87 | games=1000 | win_rate=0.00 | current_time=2026-01-13 01:07:30 | duration=550.70s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 07 | mean_reward=4.71 | games=1000 | win_rate=0.00 | current_time=2026-01-13 01:16:43 | duration=552.43s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 08 | mean_reward=8.89 | games=1000 | win_rate=0.00 | current_time=2026-01-13 01:25:58 | duration=555.48s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 09 | mean_reward=10.61 | games=1000 | win_rate=0.00 | current_time=2026-01-13 01:35:15 | duration=556.27s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0010.pth
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 10 | mean_reward=3.54 | games=1000 | win_rate=0.00 | current_time=2026-01-13 01:44:26 | duration=551.01s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 11 | mean_reward=4.71 | games=1000 | win_rate=0.00 | current_time=2026-01-13 01:53:47 | duration=561.05s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 12 | mean_reward=7.73 | games=1000 | win_rate=0.00 | current_time=2026-01-13 02:03:07 | duration=560.13s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 13 | mean_reward=7.88 | games=1000 | win_rate=0.00 | current_time=2026-01-13 02:12:20 | duration=553.27s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 14 | mean_reward=5.74 | games=1000 | win_rate=0.00 | current_time=2026-01-13 02:21:42 | duration=561.91s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 15 | mean_reward=7.36 | games=1000 | win_rate=0.00 | current_time=2026-01-13 02:30:53 | duration=550.49s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 16 | mean_reward=6.35 | games=1000 | win_rate=0.00 | current_time=2026-01-13 02:40:12 | duration=558.65s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 17 | mean_reward=4.24 | games=1000 | win_rate=0.00 | current_time=2026-01-13 02:49:21 | duration=549.21s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 18 | mean_reward=5.86 | games=1000 | win_rate=0.00 | current_time=2026-01-13 02:58:34 | duration=553.04s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 19 | mean_reward=4.85 | games=1000 | win_rate=0.00 | current_time=2026-01-13 03:07:52 | duration=557.80s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0020.pth
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 20 | mean_reward=4.91 | games=1000 | win_rate=0.00 | current_time=2026-01-13 03:17:10 | duration=557.77s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 21 | mean_reward=6.58 | games=1000 | win_rate=0.00 | current_time=2026-01-13 03:26:32 | duration=561.83s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 22 | mean_reward=8.30 | games=1000 | win_rate=0.00 | current_time=2026-01-13 03:35:47 | duration=555.49s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 23 | mean_reward=9.08 | games=1000 | win_rate=0.00 | current_time=2026-01-13 03:45:01 | duration=553.38s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 24 | mean_reward=7.62 | games=1000 | win_rate=0.00 | current_time=2026-01-13 03:54:19 | duration=558.18s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 25 | mean_reward=6.78 | games=1000 | win_rate=0.00 | current_time=2026-01-13 04:03:39 | duration=559.29s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 26 | mean_reward=1.57 | games=1000 | win_rate=0.00 | current_time=2026-01-13 04:12:57 | duration=558.75s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 27 | mean_reward=7.55 | games=1000 | win_rate=0.00 | current_time=2026-01-13 04:22:11 | duration=553.58s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 28 | mean_reward=6.43 | games=1000 | win_rate=0.00 | current_time=2026-01-13 04:31:24 | duration=552.47s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 29 | mean_reward=1.95 | games=1000 | win_rate=0.00 | current_time=2026-01-13 04:40:43 | duration=559.34s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0030.pth
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 30 | mean_reward=2.58 | games=1000 | win_rate=0.00 | current_time=2026-01-13 04:49:57 | duration=553.99s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 31 | mean_reward=3.07 | games=1000 | win_rate=0.00 | current_time=2026-01-13 04:59:17 | duration=559.39s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 32 | mean_reward=6.95 | games=1000 | win_rate=0.00 | current_time=2026-01-13 05:08:32 | duration=555.48s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 33 | mean_reward=7.18 | games=1000 | win_rate=0.00 | current_time=2026-01-13 05:17:48 | duration=555.45s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 34 | mean_reward=6.72 | games=1000 | win_rate=0.00 | current_time=2026-01-13 05:27:03 | duration=555.29s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 35 | mean_reward=3.24 | games=1000 | win_rate=0.00 | current_time=2026-01-13 05:36:21 | duration=558.11s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 36 | mean_reward=4.10 | games=1000 | win_rate=0.00 | current_time=2026-01-13 05:45:34 | duration=552.04s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 37 | mean_reward=3.65 | games=1000 | win_rate=0.00 | current_time=2026-01-13 05:54:46 | duration=552.67s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 38 | mean_reward=5.12 | games=1000 | win_rate=0.00 | current_time=2026-01-13 06:04:03 | duration=556.54s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 39 | mean_reward=5.69 | games=1000 | win_rate=0.00 | current_time=2026-01-13 06:13:21 | duration=557.58s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0040.pth
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 40 | mean_reward=6.86 | games=1000 | win_rate=0.00 | current_time=2026-01-13 06:22:35 | duration=554.45s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 41 | mean_reward=3.34 | games=1000 | win_rate=0.00 | current_time=2026-01-13 06:31:53 | duration=557.83s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 42 | mean_reward=6.66 | games=1000 | win_rate=0.00 | current_time=2026-01-13 06:41:08 | duration=554.35s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 43 | mean_reward=7.43 | games=1000 | win_rate=0.00 | current_time=2026-01-13 06:50:19 | duration=550.97s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 44 | mean_reward=8.39 | games=1000 | win_rate=0.00 | current_time=2026-01-13 06:59:31 | duration=551.75s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 45 | mean_reward=8.36 | games=1000 | win_rate=0.00 | current_time=2026-01-13 07:08:47 | duration=556.74s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 46 | mean_reward=3.31 | games=1000 | win_rate=0.00 | current_time=2026-01-13 07:18:06 | duration=558.89s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 47 | mean_reward=5.81 | games=1000 | win_rate=0.00 | current_time=2026-01-13 07:27:24 | duration=557.35s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 48 | mean_reward=4.91 | games=1000 | win_rate=0.00 | current_time=2026-01-13 07:36:44 | duration=560.40s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 49 | mean_reward=5.84 | games=1000 | win_rate=0.00 | current_time=2026-01-13 07:45:59 | duration=554.99s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0050.pth
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 50 | mean_reward=4.24 | games=1000 | win_rate=0.00 | current_time=2026-01-13 07:55:13 | duration=553.64s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 51 | mean_reward=10.52 | games=1000 | win_rate=0.00 | current_time=2026-01-13 08:04:24 | duration=550.70s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 52 | mean_reward=9.72 | games=1000 | win_rate=0.00 | current_time=2026-01-13 08:13:35 | duration=550.80s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 53 | mean_reward=8.09 | games=1000 | win_rate=0.00 | current_time=2026-01-13 08:22:51 | duration=556.16s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 54 | mean_reward=8.13 | games=1000 | win_rate=0.00 | current_time=2026-01-13 08:32:11 | duration=559.59s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 55 | mean_reward=8.20 | games=1000 | win_rate=0.00 | current_time=2026-01-13 08:41:25 | duration=554.02s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 56 | mean_reward=6.61 | games=1000 | win_rate=0.00 | current_time=2026-01-13 08:50:43 | duration=557.50s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 57 | mean_reward=6.67 | games=1000 | win_rate=0.00 | current_time=2026-01-13 09:00:00 | duration=557.57s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 58 | mean_reward=4.77 | games=1000 | win_rate=0.00 | current_time=2026-01-13 09:09:19 | duration=558.50s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 59 | mean_reward=4.09 | games=1000 | win_rate=0.00 | current_time=2026-01-13 09:18:34 | duration=555.02s
Saved checkpoint to checkpoints/jass_rl_agent_epoch_0060.pth
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 60 | mean_reward=6.32 | games=1000 | win_rate=0.00 | current_time=2026-01-13 09:27:46 | duration=552.44s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 61 | mean_reward=6.82 | games=1000 | win_rate=0.00 | current_time=2026-01-13 09:36:58 | duration=551.63s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 62 | mean_reward=7.07 | games=1000 | win_rate=0.00 | current_time=2026-01-13 09:46:13 | duration=554.37s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 63 | mean_reward=5.17 | games=1000 | win_rate=0.00 | current_time=2026-01-13 09:55:27 | duration=554.12s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 64 | mean_reward=2.74 | games=1000 | win_rate=0.00 | current_time=2026-01-13 10:04:45 | duration=557.95s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 65 | mean_reward=8.76 | games=1000 | win_rate=0.00 | current_time=2026-01-13 10:13:54 | duration=549.16s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device
Epoch 66 | mean_reward=6.99 | games=1000 | win_rate=0.00 | current_time=2026-01-13 10:23:08 | duration=553.89s
MCTS_Observation_GPU uses cuda as device
MCTS_Observation_GPU uses cuda as device


#### LOGS Playouts von RL Agent gegen MCTS Observation Agent und Rule Based Agent
jovyan@jupyter-fabian-stettler---af06958b:~/DL4G/JassAgent$ python test_arena/test_strategy_comparison.py 
2026-01-13 10:38:52,969 INFO Attempting to load RL agent weights from /home/jovyan/DL4G/JassAgent/weights_rl_agent/jass_rl_agent_epoch_0050.pth
2026-01-13 10:38:52,977 INFO Loaded RL agent weights from /home/jovyan/DL4G/JassAgent/weights_rl_agent/jass_rl_agent_epoch_0050.pth
Using RL agent weights from: /home/jovyan/DL4G/JassAgent/weights_rl_agent/jass_rl_agent_epoch_0050.pth
MCTS_Observation_GPU uses None as device
MCTS_Observation_GPU uses None as device


🔍 Matchup: MCTS Observation vs RL Agent
--------------------------------------------------
Average Points (RL vs Opp): 81.0 vs 76.0
Advantage: 5.0 points
Win Rate: 60.0%
Time per game: 58.07s


🔍 Matchup: Rule Based vs RL Agent
--------------------------------------------------
Average Points (RL vs Opp): 81.6 vs 75.4
Advantage: 6.2 points
Win Rate: 50.0%
Time per game: 0.02s

============================================================
📊 SUMMARY
============================================================
Matchup                    Advantage     Win Rate    Time/Game
------------------------------------------------------------
MCTS Observation vs RL Agent        5.0       60.0%       58.07s
Rule Based                       6.2       50.0%        0.02s

🏆 Highest Advantage: Rule Based (+6.2 points) 