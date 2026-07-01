import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from env import AtariEnv
from agent import DQNAgent
import config

GAME           = config.GAME
FRAME_STACK    = config.FRAME_STACK
CHECKPOINT_DIR = config.CHECKPOINT_DIR
VIDEO_DIR      = config.VIDEO_DIR


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
    Runs until the game actually ends (all lives lost), not on life loss.
    """
    import cv2

    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = AtariEnv(GAME, frame_stack=FRAME_STACK, clip_rewards=False, render_mode="rgb_array")

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        frames = []

        while True:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, info = env.step(action)
            frames.append(env.render())
            total_reward += reward
            if info.get('real_done', done):  # stop only when all lives are gone
                break

        # write frames to mp4
        video_path = os.path.join(VIDEO_DIR, f"episode_{ep + 1}.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        print(f"recorded episode {ep + 1} | reward {total_reward:.1f} | frames {len(frames)}")

    env.close()
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
