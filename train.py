import os
import torch
import numpy as np
import psutil
import wandb
from env import AtariEnv
from replay_buffer import ReplayBuffer
from agent import DQNAgent
import config


def get_epsilon(step: int) -> float:
    """linearly decay epsilon from EPSILON_START to EPSILON_END over EPSILON_DECAY steps"""
    res = config.EPSILON_START + (config.EPSILON_END - config.EPSILON_START) * (step / config.EPSILON_DECAY)
    return max(config.EPSILON_END, res)


def save_checkpoint(agent: DQNAgent, total_steps: int, episode: int):
    """saves online network weights and training state to disk"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(config.CHECKPOINT_DIR, f"dqn_step_{total_steps}.pt")
    torch.save({
        "step":                 total_steps,
        "episode":              episode,
        "online_state_dict":    agent.online_net.state_dict(),
        "target_state_dict":    agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }, path)
    print(f"checkpoint saved → {path}")


def train():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"using device: {device}")

    wandb.init(
        project="atari-dqn",
        config={
            "game":             config.GAME,
            "max_steps":        config.MAX_STEPS,
            "replay_size":      config.REPLAY_SIZE,
            "batch_size":       config.BATCH_SIZE,
            "learning_rate":    config.LEARNING_RATE,
            "gamma":            config.GAMMA,
            "epsilon_start":    config.EPSILON_START,
            "epsilon_end":      config.EPSILON_END,
            "epsilon_decay":    config.EPSILON_DECAY,
            "min_replay_size":  config.MIN_REPLAY_SIZE,
            "target_sync_freq": config.TARGET_SYNC_FREQ,
        }
    )

    env    = AtariEnv(config.GAME, frame_stack=config.FRAME_STACK)
    buffer = ReplayBuffer(max_size=config.REPLAY_SIZE, batch_size=config.BATCH_SIZE)
    agent  = DQNAgent(
        in_channels=config.FRAME_STACK,
        n_actions=env.action_space,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        device=device
    )

    total_steps     = 0
    episode         = 0
    last_checkpoint = 0

    while total_steps < config.MAX_STEPS:
        state     = env.reset()
        ep_reward = 0
        ep_loss   = []
        done      = False

        while not done:
            epsilon = get_epsilon(total_steps)

            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            buffer.push(state, action, reward, next_state, float(done))

            state       = next_state
            ep_reward  += reward
            total_steps += 1

            if len(buffer) >= config.MIN_REPLAY_SIZE:
                s, a, r, s_next, d = buffer.sample()
                loss = agent.train_step(s, a, r, s_next, d)
                ep_loss.append(loss)

            if total_steps % config.TARGET_SYNC_FREQ == 0:
                agent.sync_target()

            if total_steps - last_checkpoint >= config.CHECKPOINT_FREQ:
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
            "episode":     episode,
            "reward":      ep_reward,
            "epsilon":     epsilon,
            "cpu_ram_gb":  ram_used,
            "gpu_ram_gb":  gpu_mem,
            "total_steps": total_steps,
        }
        if mean_loss is not None:
            log["loss"] = mean_loss

        if len(buffer) >= config.MIN_REPLAY_SIZE:
            s, _, _, _, _ = buffer.sample()
            s_tensor = torch.tensor(s, dtype=torch.float32).to(agent.device)
            with torch.no_grad():
                q_values = agent.online_net(s_tensor)
            log["mean_max_q"] = q_values.max(dim=1).values.mean().item()

        wandb.log(log, step=total_steps)

    save_checkpoint(agent, total_steps, episode)
    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
