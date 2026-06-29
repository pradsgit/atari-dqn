import os
import torch
import numpy as np
import psutil
import wandb
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
CHECKPOINT_FREQ  = 100_000        # steps between checkpoints
CHECKPOINT_DIR   = "/content/drive/MyDrive/atari-dqn/checkpoints"


def get_epsilon(step: int) -> float:
    """linearly decay epsilon from EPSILON_START to EPSILON_END over EPSILON_DECAY steps"""
    res = EPSILON_START + (EPSILON_END - EPSILON_START) * (step / EPSILON_DECAY)
    return max(EPSILON_END, res)


def save_checkpoint(agent: DQNAgent, total_steps: int, episode: int):
    """saves online network weights and training state to disk"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"dqn_step_{total_steps}.pt")
    torch.save({
        "step":             total_steps,
        "episode":          episode,
        "online_state_dict": agent.online_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }, path)
    print(f"checkpoint saved → {path}")


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    wandb.init(
        project="atari-dqn",
        config={
            "game":             GAME,
            "max_steps":        MAX_STEPS,
            "replay_size":      REPLAY_SIZE,
            "batch_size":       BATCH_SIZE,
            "learning_rate":    LEARNING_RATE,
            "gamma":            GAMMA,
            "epsilon_start":    EPSILON_START,
            "epsilon_end":      EPSILON_END,
            "epsilon_decay":    EPSILON_DECAY,
            "min_replay_size":  MIN_REPLAY_SIZE,
            "target_sync_freq": TARGET_SYNC_FREQ,
        }
    )

    env    = AtariEnv(GAME, frame_stack=FRAME_STACK)
    buffer = ReplayBuffer(max_size=REPLAY_SIZE, batch_size=BATCH_SIZE)
    agent  = DQNAgent(
        in_channels=FRAME_STACK,
        n_actions=env.action_space,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        device=device
    )

    total_steps      = 0
    episode          = 0
    last_checkpoint  = 0
    ep_loss          = []

    while total_steps < MAX_STEPS:
        state     = env.reset()
        ep_reward = 0
        ep_loss   = []
        done      = False

        while not done:
            epsilon = get_epsilon(total_steps)

            # select and execute action
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # store transition
            buffer.push(state, action, reward, next_state, float(done))

            state       = next_state
            ep_reward  += reward
            total_steps += 1

            # train once buffer is large enough
            if len(buffer) >= MIN_REPLAY_SIZE:
                s, a, r, s_next, d = buffer.sample()
                loss = agent.train_step(s, a, r, s_next, d)
                ep_loss.append(loss)

            # sync target network
            if total_steps % TARGET_SYNC_FREQ == 0:
                agent.sync_target()

            # save checkpoint
            if total_steps - last_checkpoint >= CHECKPOINT_FREQ:
                save_checkpoint(agent, total_steps, episode)
                last_checkpoint = total_steps

        episode  += 1
        ram_used  = psutil.virtual_memory().used / 1e9
        gpu_mem   = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        epsilon   = get_epsilon(total_steps)
        mean_loss = np.mean(ep_loss) if ep_loss else None
        loss_str  = f"{mean_loss:.4f}" if mean_loss is not None else "collecting"
        print(f"episode {episode:4d} | steps {total_steps:8d} | reward {ep_reward:.1f} | epsilon {epsilon:.3f} | loss {loss_str} | cpu {ram_used:.1f}GB | gpu {gpu_mem:.1f}GB")

        log = {
            "episode":          episode,
            "reward":           ep_reward,
            "epsilon":          epsilon,
            "cpu_ram_gb":       ram_used,
            "gpu_ram_gb":       gpu_mem,
            "total_steps":      total_steps,
        }
        if mean_loss is not None:
            log["loss"] = mean_loss

        # mean max Q-value over a sample batch — tracks whether Q-values are growing or collapsing
        if len(buffer) >= MIN_REPLAY_SIZE:
            s, _, _, _, _ = buffer.sample()
            s_tensor = torch.tensor(s, dtype=torch.float32).to(agent.device)
            with torch.no_grad():
                q_values = agent.online_net(s_tensor)
            log["mean_max_q"] = q_values.max(dim=1).values.mean().item()

        wandb.log(log, step=total_steps)

    # save final checkpoint
    save_checkpoint(agent, total_steps, episode)
    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
