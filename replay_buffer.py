import numpy as np


class ReplayBuffer:
    """
    Fixed-size ring buffer storing (s, a, r, s', done) transitions.
    States stored as uint8 to save 4x memory vs float32.
    Uses numpy arrays for O(1) indexing and fast batch sampling.
    """
    def __init__(self, max_size: int = 1_000_000, batch_size: int = 32):
        self.max_size   = max_size
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0

        # pre-allocate arrays — shape filled in on first push
        self._states      = None
        self._next_states = None
        self._actions     = None
        self._rewards     = None
        self._dones       = None

    def _init_arrays(self, state_shape):
        self._states      = np.zeros((self.max_size, *state_shape), dtype=np.uint8)
        self._next_states = np.zeros((self.max_size, *state_shape), dtype=np.uint8)
        self._actions     = np.zeros(self.max_size, dtype=np.int64)
        self._rewards     = np.zeros(self.max_size, dtype=np.float32)
        self._dones       = np.zeros(self.max_size, dtype=np.float32)

    def push(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        if self._states is None:
            self._init_arrays(s.shape)

        self._states[self.ptr]      = (s * 255).astype(np.uint8)
        self._next_states[self.ptr] = (s_next * 255).astype(np.uint8)
        self._actions[self.ptr]     = a
        self._rewards[self.ptr]     = r
        self._dones[self.ptr]       = float(done)

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        assert self.size >= self.batch_size, "not enough transitions in buffer to sample"
        idx = np.random.randint(0, self.size, size=self.batch_size)
        return (
            self._states[idx].astype(np.float32)      / 255.0,
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx].astype(np.float32) / 255.0,
            self._dones[idx],
        )

    def __len__(self):
        return self.size
