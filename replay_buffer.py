from collections import deque
import numpy as np
import random

class ReplayBuffer:
    """
    saves state transitions in the form (s, a, r, s`, done)
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

    def push(self, s: np.array, a: int, r: float, s_next: np.array, done: bool):
        """
        stores one transition into the buffer

        Args:
            s: current observation of the game
            a: action taken in this state
            r: reward obtained
            s_next: next observation of the game
            done: is the game done?
        """

        self.buffer.append((s, a, r, s_next, done))

    def sample(self):
        """
        randomly pick a minibatch from the buffer and return it
        """
        assert len(self.buffer) >= self.batch_size, "not enough transitions in buffer to sample"
        temp = random.sample(self.buffer, self.batch_size)

        # returns list of tuples. how would you unpack this?
        s, a, r, s_next, done = zip(*temp) # zip(*temp) transposes the list of tuples — turns [(s1,a1,...), (s2,a2,...)] into [(s1,s2,...), (a1,a2,...)].
        return np.array(s), np.array(a), np.array(r), np.array(s_next), np.array(done)


        