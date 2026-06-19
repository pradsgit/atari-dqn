import torch
import numpy as np
import psutil
from env import AtariEnv
from replay_buffer import ReplayBuffer
from agent import DQNAgent


# --- hyperparameters ---
GAME             = "Breakout"
FRAME_STACK      = 4
MAX_EPISODES     = 10_000
MAX_STEPS        = 50_000_000     # paper trains for 50M frames
REPLAY_SIZE      = 1_000_000      # replay buffer capacity
BATCH_SIZE       = 32
LEARNING_RATE    = 0.00025        # from paper
GAMMA            = 0.99           # discount factor
EPSILON_START    = 1.0            # initial exploration rate
EPSILON_END      = 0.1            # final exploration rate
EPSILON_DECAY    = 1_000_000      # steps to decay epsilon over
MIN_REPLAY_SIZE  = 50_000         # steps before training starts
TARGET_SYNC_FREQ = 10_000         # steps between target network syncs


def get_epsilon(step: int) -> float:
    """linearly decay epsilon from EPSILON_START to EPSILON_END over EPSILON_DECAY steps"""
    res = EPSILON_START + (EPSILON_END - EPSILON_START) * (step / EPSILON_DECAY)
    return max(EPSILON_END, res)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    env    = AtariEnv(GAME, frame_stack=FRAME_STACK)
    buffer = ReplayBuffer(max_size=REPLAY_SIZE, batch_size=BATCH_SIZE)
    agent  = DQNAgent(
        in_channels=FRAME_STACK,
        n_actions=env.action_space,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        device=device
    )

    total_steps   = 0
    episode       = 0

    while total_steps < MAX_STEPS:
        state   = env.reset()
        ep_reward = 0
        done    = False

        while not done:
            epsilon = get_epsilon(total_steps)

            # select and execute action
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # store transition
            buffer.push(state, action, reward, next_state, float(done))

            state      = next_state
            ep_reward += reward
            total_steps += 1

            # train once buffer is large enough
            if len(buffer) >= MIN_REPLAY_SIZE:
                s, a, r, s_next, d = buffer.sample()
                loss = agent.train_step(s, a, r, s_next, d)

            # sync target network
            if total_steps % TARGET_SYNC_FREQ == 0:
                agent.sync_target()

        episode += 1
        ram_used = psutil.virtual_memory().used / 1e9
        print(f"episode {episode:4d} | steps {total_steps:8d} | reward {ep_reward:.1f} | epsilon {get_epsilon(total_steps):.3f} | ram {ram_used:.1f}GB")

    env.close()


if __name__ == "__main__":
    train()
