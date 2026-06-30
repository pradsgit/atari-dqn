import os
import torch
import numpy as np
import psutil
import wandb
from vec_env import VecAtariEnv
from replay_buffer import ReplayBuffer
from agent import DQNAgent
import config


def get_epsilon(step: int) -> float:
    res = config.EPSILON_START + (config.EPSILON_END - config.EPSILON_START) * (step / config.EPSILON_DECAY)
    return max(config.EPSILON_END, res)


def save_checkpoint(agent: DQNAgent, total_steps: int, episode: int):
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
    print(f"using device: {device}, n_envs: {config.N_ENVS}")

    wandb.init(
        project="atari-dqn",
        config={
            "game":             config.GAME,
            "n_envs":           config.N_ENVS,
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

    vec_env = VecAtariEnv(config.GAME, n_envs=config.N_ENVS, frame_stack=config.FRAME_STACK)
    buffer  = ReplayBuffer(max_size=config.REPLAY_SIZE, batch_size=config.BATCH_SIZE)
    agent   = DQNAgent(
        in_channels=config.FRAME_STACK,
        n_actions=vec_env.action_space,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        device=device
    )

    total_steps      = 0
    episode          = 0
    last_checkpoint  = 0
    last_target_sync = 0

    # per-env episode tracking
    ep_rewards = np.zeros(config.N_ENVS)
    ep_losses  = [[] for _ in range(config.N_ENVS)]

    states = vec_env.reset()

    while total_steps < config.MAX_STEPS:
        epsilon = get_epsilon(total_steps)

        # select actions for all envs
        actions = np.array([agent.select_action(states[i], epsilon) for i in range(config.N_ENVS)])

        # step all envs in parallel
        next_states, rewards, dones, _ = vec_env.step(actions)

        # store N transitions
        for i in range(config.N_ENVS):
            buffer.push(states[i], actions[i], rewards[i], next_states[i], float(dones[i]))
            ep_rewards[i] += rewards[i]
            total_steps   += 1

        states = next_states

        # train once per vectorized step
        if len(buffer) >= config.MIN_REPLAY_SIZE:
            s, a, r, s_next, d = buffer.sample()
            loss = agent.train_step(s, a, r, s_next, d)
            for i in range(config.N_ENVS):
                ep_losses[i].append(loss)

        # sync target network
        if total_steps - last_target_sync >= config.TARGET_SYNC_FREQ:
            agent.sync_target()
            last_target_sync = total_steps

        # checkpoint
        if total_steps - last_checkpoint >= config.CHECKPOINT_FREQ:
            save_checkpoint(agent, total_steps, episode)
            last_checkpoint = total_steps

        # log completed episodes
        for i in range(config.N_ENVS):
            if dones[i]:
                episode   += 1
                ram_used   = psutil.virtual_memory().used / 1e9
                gpu_mem    = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                mean_loss  = np.mean(ep_losses[i]) if ep_losses[i] else None
                loss_str   = f"{mean_loss:.4f}" if mean_loss is not None else "collecting"
                print(f"episode {episode:4d} | steps {total_steps:8d} | reward {ep_rewards[i]:.1f} | epsilon {epsilon:.3f} | loss {loss_str} | cpu {ram_used:.1f}GB | gpu {gpu_mem:.1f}GB")

                log = {
                    "episode":     episode,
                    "reward":      ep_rewards[i],
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

                ep_rewards[i] = 0.0
                ep_losses[i]  = []

    save_checkpoint(agent, total_steps, episode)
    vec_env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
