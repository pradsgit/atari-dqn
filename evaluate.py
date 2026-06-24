import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from env import AtariEnv
from agent import DQNAgent
from gymnasium.wrappers import RecordVideo


GAME        = "Breakout"
FRAME_STACK = 4
CHECKPOINT_DIR = "/content/drive/MyDrive/atari-dqn/checkpoints"
VIDEO_DIR      = "/content/drive/MyDrive/atari-dqn/videos"


def load_agent(checkpoint_path: str, device: str) -> DQNAgent:
    """loads a trained agent from a checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make(f"ALE/{GAME}-v5")
    n_actions = env.action_space.n
    env.close()

    agent = DQNAgent(
        in_channels=FRAME_STACK,
        n_actions=n_actions,
        lr=1e-4,
        gamma=0.99,
        device=device
    )
    agent.online_net.load_state_dict(checkpoint["online_state_dict"])
    agent.online_net.eval()

    print(f"loaded checkpoint from step {checkpoint['step']}, episode {checkpoint['episode']}")
    return agent


def evaluate(agent: DQNAgent, n_episodes: int = 10) -> tuple[float, float]:
    """
    runs n_episodes with epsilon=0 (pure exploitation) and returns
    mean and std of episode rewards.

    Args:
        agent: trained DQNAgent
        n_episodes: number of episodes to evaluate over

    Returns:
        mean reward, std reward
    """
    env = AtariEnv(GAME, frame_stack=FRAME_STACK, clip_rewards=False)
    rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"eval episode {ep+1:3d} | reward {total_reward:.1f}")

    env.close()
    mean, std = np.mean(rewards), np.std(rewards)
    print(f"\nmean reward: {mean:.1f} ± {std:.1f} over {n_episodes} episodes")
    return mean, std


def record_video(agent: DQNAgent, n_episodes: int = 1):
    """
    records gameplay video of the agent playing with epsilon=0.
    Uses gymnasium RecordVideo wrapper around the raw ALE env for rendering,
    while using AtariEnv for preprocessing and action selection.

    Args:
        agent: trained DQNAgent
        n_episodes: number of episodes to record
    """
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)

    os.makedirs(VIDEO_DIR, exist_ok=True)

    # raw env for video recording
    raw_env = gym.make(f"ALE/{GAME}-v5", frameskip=4, render_mode="rgb_array")
    raw_env = RecordVideo(raw_env, video_folder=VIDEO_DIR, episode_trigger=lambda e: True)

    # preprocessed env for agent
    atari_env = AtariEnv(GAME, frame_stack=FRAME_STACK, clip_rewards=False)

    for ep in range(n_episodes):
        raw_env.reset()
        state = atari_env.reset()
        done  = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, _ = atari_env.step(action)
            raw_env.step(action)   # advances raw env for video recording
            total_reward += reward

        print(f"recorded episode {ep+1} | reward {total_reward:.1f}")

    raw_env.close()
    atari_env.close()
    print(f"videos saved to {VIDEO_DIR}")


def plot_training_curve(log_file: str):
    """
    plots reward over episodes from a training log file.
    expects one line per episode with 'reward X.X' in it.

    Args:
        log_file: path to training log file
    """
    rewards = []
    steps   = []

    with open(log_file) as f:
        for line in f:
            if "reward" in line and "episode" in line:
                parts = {p.split()[0]: p.split()[1] for p in line.strip().split("|")}
                try:
                    rewards.append(float(parts["reward"]))
                    steps.append(int(parts["steps"]))
                except (KeyError, ValueError):
                    continue

    # smooth with rolling average
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(12, 5))
    plt.plot(steps[:len(smoothed)], smoothed, label=f"reward (rolling avg {window})")
    plt.xlabel("steps")
    plt.ylabel("episode reward")
    plt.title(f"DQN Training Curve — {GAME}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "training_curve.png"))
    plt.show()
    print("training curve saved")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # find latest checkpoint
    checkpoints = sorted([
        f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")
    ])
    if not checkpoints:
        print("no checkpoints found")
    else:
        latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        agent = load_agent(latest, device)
        evaluate(agent, n_episodes=10)
        record_video(agent, n_episodes=1)
