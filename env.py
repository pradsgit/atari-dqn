import numpy as np
import cv2
import gymnasium as gym
from collections import deque


class AtariEnv:
    """
    Wraps an ALE Atari env with the preprocessing from the DQN 2015 paper:
    - Pixel-wise max over last 2 frames to remove sprite flickering
    - Extract Y channel (luminance) and resize to 84x84
    - Stack 4 consecutive processed frames as the state
    - Clip rewards to {-1, 0, 1}
    """

    def __init__(self, game: str, frame_stack: int = 4, clip_rewards: bool = True):
        """
        Args:
            game: Atari game name (e.g. 'Breakout', 'Pong')
            frame_stack: number of consecutive frames to stack as one state
            clip_rewards: whether to clip rewards to {-1, 0, 1} via np.sign
        """
        import ale_py

        gym.register_envs(ale_py)

        self.env = gym.make(f"ALE/{game}-v5", frameskip=1)
        self.frame_stack = frame_stack
        self.clip_rewards = clip_rewards
        self.frames = deque(maxlen=frame_stack)
        self.prev_frame = None  # needed for pixel-wise max

        self.observation_space = (frame_stack, 84, 84)
        self.action_space = self.env.action_space.n

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies DQN preprocessing to a single raw RGB frame.

        Steps:
          1. Pixel-wise max with previous frame — removes sprite flickering
             caused by Atari 2600 hardware alternating sprites across frames.
          2. Convert RGB to YCrCb and extract Y (luminance) channel — reduces
             3 color channels to 1 brightness channel.
          3. Resize to 84x84 and normalize to [0, 1].

        Args:
            frame: raw RGB frame from the emulator, shape (210, 160, 3)

        Returns:
            processed frame of shape (84, 84), dtype float32, values in [0, 1]
        """
        if self.prev_frame is not None:
            frame = np.maximum(frame, self.prev_frame)
        self.prev_frame = frame

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:, :, 0]

        resized = cv2.resize(y_channel, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def reset(self) -> np.ndarray:
        """
        Resets the environment and returns the initial state.

        Fills the frame stack with copies of the first processed frame so the
        agent always receives a full (frame_stack, 84, 84) state from the start.

        Returns:
            initial state of shape (frame_stack, 84, 84), dtype float32
        """
        frame, _ = self.env.reset()
        self.prev_frame = None
        processed = self._preprocess(frame)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return self._get_state()

    def step(self, action: int):
        """
        Executes one action in the environment.

        Args:
            action: integer index into the action space

        Returns:
            state:  next state of shape (frame_stack, 84, 84), dtype float32
            reward: clipped to {-1, 0, 1} if clip_rewards=True
            done:   True if the episode has ended
            info:   dict of auxiliary info from the emulator
        """
        frame, reward, terminated, truncated, info = self.env.step(action)
        processed = self._preprocess(frame)
        self.frames.append(processed)

        if self.clip_rewards:
            reward = np.sign(reward)

        done = terminated or truncated
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        Returns the current stacked frame state.

        Returns:
            array of shape (frame_stack, 84, 84), dtype float32
        """
        return np.array(self.frames, dtype=np.float32)

    def close(self):
        """Closes the underlying environment and releases resources."""
        self.env.close()
