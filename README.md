# Atari DQN

PyTorch implementation of [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2015), trained on Breakout.

<video src="videos/gameplay.mp4" controls width="400"></video>

Agent reaching 80+ reward after 10M steps of training.

---

## Results

| Steps | Mean Reward |
|-------|-------------|
| 1M    | ~5          |
| 3M    | ~50         |
| 10M   | ~60-80      |

Human-level performance on Breakout is ~31. The agent consistently exceeds this after ~2M steps.

---

## Architecture

Standard DQN from the 2015 paper:

- **CNN**: 3 conv layers → fully connected → Q-value per action
- **Experience replay**: 600k transition ring buffer (uint8 storage)
- **Target network**: frozen copy of online network, synced every 10k steps
- **Epsilon-greedy**: linear decay from 1.0 → 0.1 over 1M steps

Key preprocessing matching the paper:
- Grayscale + resize to 84×84
- Stack 4 consecutive frames as state
- Max-pool last 2 raw frames per skip to remove sprite flickering
- `repeat_action_probability=0.0` (no sticky actions)
- Episodic-life termination: `done=True` on life loss for TD bootstrapping

Training uses 12 parallel environments via `multiprocessing` for ~4× wall-clock speedup, with 3 gradient updates per outer loop to match the paper's 1:4 update-to-step ratio.

---

## Project Structure

```
env.py            # Atari env wrapper with DQN preprocessing
model.py          # CNN architecture
agent.py          # DQNAgent: action selection, train step, target sync
replay_buffer.py  # Ring buffer with uint8 state storage
vec_env.py        # Vectorized envs via multiprocessing
train.py          # Training loop
dqn_evaluate.py   # Evaluation and video recording
config.py         # Hyperparameters
run_training.ipynb # Colab training notebook
```

---

## Setup

```bash
uv sync
```

## Train

```bash
uv run python train.py
```

Override config for Colab in `run_training.ipynb` cell 10.

## Evaluate

```python
from dqn_evaluate import load_agent, evaluate, record_video

agent = load_agent("checkpoints/dqn_step_10000000.pt", device="cuda")
evaluate(agent, n_episodes=10)
record_video(agent, n_episodes=1)
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Replay buffer | 600k |
| Batch size | 32 |
| Learning rate | 0.00025 |
| Gamma | 0.99 |
| Epsilon decay | 1M steps |
| Target sync | every 10k steps |
| Frameskip | 4 |
| Frame stack | 4 |
