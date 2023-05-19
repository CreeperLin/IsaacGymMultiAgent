# IsaacGymMultiAgent

> Supercharged Isaac Gym environments with multi-agent and multi-algorithm support

## Requirements

- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)

Optional:

- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [rl_games](https://github.com/Denys88/rl_games)
- tianshou
- stable_baseline3

## Installation

```bash
cd IsaacGymMultiAgent
pip3 install -e .
```

## Usage

### Training

Run IsaacGymEnvs experiments (rl_games PPO implementation)

```bash
cd examples/original
python3 train.py task=Ant
```

Run legged_gym experiments (rsl_rl PPO implementation)

```bash
cd examples/rsl_rl
python3 train.py task=a1
```

Run legged_gym environments with rl_games PPO

```bash
cd examples/original
python3 train.py task=LeggedGymA1
```

Run IsaacGym environments with rsl_rl PPO

```bash
cd examples/rsl_rl
python3 train.py task=Ant
```

Run IsaacGym environments with stable-baselines3 PPO

```bash
cd examples/sb3
python3 train.py task=Ant train=sb3ppo
```

### Playing

```bash
cd examples/original
python3 train.py task=Ant test=True headless=False checkpoint=path/to/checkpoint.pt
```

### Rendering

Supported displays:
- VNC: Tested on ```tigervnc-standalone-server```
- Alternatives: Xvfb, VirtualGL, etc.

Add ```headless=False``` to enable rendering

```bash
export DISPLAY=:1
python train.py ... headless=False
```

## List of Environments

| Name                  | Description                                        |
| --------------------- | -------------------------------------------------- |
| LeggedGymA1           | wrapped legged_gym environment for Unitree A1 (a1) |
| LeggedGymAnymalCRough | wrapped legged_gym env (anymal_c_rough)            |
| LeggedGymAnymalCFlat  | wrapped legged_gym env (anymal_c_flat)             |
| LeggedGymAnymalB      | wrapped legged_gym env (anymal_b)                  |
| LeggedGymCassie       | wrapped legged_gym env (cassie)                    |
| BallJoust             |                                                    |
| QuadcopterJoust       |                                                    |

## Troubleshooting

> ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory

Add the python library path to LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/python/lib
```
