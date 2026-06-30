import numpy as np
import multiprocessing as mp
from env import AtariEnv


def _worker(remote, parent_remote, game, frame_stack, clip_rewards, frameskip):
    """runs in a subprocess — owns one AtariEnv, responds to commands from main process"""
    parent_remote.close()
    env = AtariEnv(game, frame_stack=frame_stack, clip_rewards=clip_rewards, frameskip=frameskip)

    while True:
        cmd, data = remote.recv()
        if cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'step':
            state, reward, done, info = env.step(data)
            if info.get('real_done', done):
                state = env.reset()
            remote.send((state, reward, done, info))
        elif cmd == 'close':
            env.close()
            remote.close()
            break


class VecAtariEnv:
    """
    Runs N AtariEnv instances in parallel subprocesses.
    reset() and step() return batched arrays of shape (N, ...).
    Done envs are auto-reset so the main loop never needs to reset manually.
    """

    def __init__(self, game: str, n_envs: int, frame_stack: int = 4,
                 clip_rewards: bool = True, frameskip: int = 4):
        self.n_envs = n_envs

        # get action space from a temporary env
        _tmp = AtariEnv(game, frame_stack=frame_stack)
        self.action_space = _tmp.action_space
        _tmp.close()

        ctx = mp.get_context('fork')
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])

        self.processes = []
        for work_remote, remote in zip(work_remotes, self.remotes):
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote, game, frame_stack, clip_rewards, frameskip),
                daemon=True
            )
            p.start()
            self.processes.append(p)

        for wr in work_remotes:
            wr.close()

    def reset(self) -> np.ndarray:
        """resets all envs, returns states of shape (N, frame_stack, 84, 84)"""
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, actions: np.ndarray):
        """
        sends one action per env, returns batched results.

        Returns:
            states:  (N, frame_stack, 84, 84)
            rewards: (N,)
            dones:   (N,)
            infos:   list of N dicts
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', int(action)))
        results = [remote.recv() for remote in self.remotes]
        states, rewards, dones, infos = zip(*results)
        return (
            np.stack(states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
