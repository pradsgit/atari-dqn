from collections import deque
import numpy as np
import random


class ReplayBuffer:
    """
    saves state transitions in the form (s, a, r, s`, done)
    States are stored as uint8 (0-255) to save memory and converted
    to float32 (0-1) on sampling.
    """
    def __init__(
        self,
        max_size: int=1_000_000,
        batch_size: int=32
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        """
        stores one transition into the buffer

        Args:
            s: current observation of the game, float32 in [0, 1]
            a: action taken in this state
            r: reward obtained
            s_next: next observation of the game, float32 in [0, 1]
            done: is the game done?
        """
        # store as uint8 to reduce memory by 4x
        self.buffer.append((
            (s * 255).astype(np.uint8),
            a,
            r,
            (s_next * 255).astype(np.uint8),
            done
        ))

    def sample(self):
        """
        randomly pick a minibatch from the buffer and return it.
        Converts states from uint8 back to float32 on sampling.
        """
        assert len(self.buffer) >= self.batch_size, "not enough transitions in buffer to sample"
        temp = random.sample(self.buffer, self.batch_size)

        s, a, r, s_next, done = zip(*temp)
        return (
            np.array(s,      dtype=np.float32) / 255.0,
            np.array(a),
            np.array(r,      dtype=np.float32),
            np.array(s_next, dtype=np.float32) / 255.0,
            np.array(done,   dtype=np.float32)
        )
