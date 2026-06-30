import numpy as np
import cv2
import gymnasium as gym
from collections import deque


class AtariEnv:
    """
    Wraps an ALE Atari env with the preprocessing from the DQN 2015 paper:
    - Manual frameskip: step the env frameskip times, max-pool the last 2 raw
      frames to remove sprite flickering (sprites alternate on consecutive frames)
    - Extract Y channel (luminance) and resize to 84x84
    - Stack 4 consecutive processed frames as the state
    - Clip rewards to {-1, 0, 1}
    - repeat_action_probability=0.0 (no sticky actions, matching the paper)
    - Episodic-life termination: done=True on life loss for TD bootstrapping,
      but the episode continues with auto-fire so the agent keeps playing
    """

    def __init__(self, game: str, frame_stack: int = 4, clip_rewards: bool = True, frameskip: int = 4, render_mode: str = None):
        import ale_py
        gym.register_envs(ale_py)

        # frameskip=1 so we control the skip loop and can max-pool raw frames
        # repeat_action_probability=0.0 matches the 2015 paper (no sticky actions)
        self.env = gym.make(f"ALE/{game}-v5", frameskip=1,
                            repeat_action_probability=0.0, render_mode=render_mode)
        self.frameskip   = frameskip
        self.frame_stack = frame_stack
        self.clip_rewards = clip_rewards
        self.frames = deque(maxlen=frame_stack)

        self.observation_space = (frame_stack, 84, 84)
        self.action_space = self.env.action_space.n
        self._lives = 0
        action_meanings = self.env.unwrapped.get_action_meanings()
        self._fire_action = action_meanings.index('FIRE') if 'FIRE' in action_meanings else None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies DQN preprocessing to a single raw RGB frame.
        Expects the frame to already be max-pooled over the last 2 emulator frames.

        Steps:
          1. Convert RGB to YCrCb and extract Y (luminance) channel
          2. Resize to 84x84 and normalize to [0, 1]
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:, :, 0]
        resized = cv2.resize(y_channel, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def reset(self) -> np.ndarray:
        frame, _ = self.env.reset()
        if self._fire_action is not None:
            frame, _, _, _, _ = self.env.step(self._fire_action)
        self._lives = self.env.unwrapped.ale.lives()
        processed = self._preprocess(frame)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return self._get_state()

    def step(self, action: int):
        """
        Executes one agent step = frameskip emulator steps.

        Max-pools the last 2 raw frames within the skip to remove sprite
        flickering (Atari alternates sprites on consecutive frames).

        Returns done=True on life loss (episodic-life termination) so the TD
        target doesn't bootstrap through a death. info['real_done'] is True
        only when the actual episode ends, used by vec_env to decide whether
        to reset.
        """
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        frame_buffer = []

        for i in range(self.frameskip):
            frame, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            # keep the last 2 raw frames for max-pooling
            if i >= self.frameskip - 2:
                frame_buffer.append(frame)
            if terminated or truncated:
                break

        if not frame_buffer:
            frame_buffer.append(frame)  # terminated before last-2 window; frame is last step's output
        max_frame = np.maximum(frame_buffer[0], frame_buffer[-1]) if len(frame_buffer) == 2 else frame_buffer[0]

        # use ale.lives() as the authoritative source — info['lives'] can lag
        current_lives = self.env.unwrapped.ale.lives()
        life_lost = current_lives < self._lives and not terminated

        if self._fire_action is not None and life_lost:
            fire_frame, _, _, _, _ = self.env.step(self._fire_action)
            max_frame = fire_frame  # update state to post-relaunch frame

        self._lives = current_lives

        processed = self._preprocess(max_frame)
        self.frames.append(processed)

        if self.clip_rewards:
            total_reward = np.sign(total_reward)

        real_done   = terminated or truncated
        done_for_td = real_done or life_lost
        info['real_done'] = real_done

        return self._get_state(), total_reward, done_for_td, info

    def _get_state(self) -> np.ndarray:
        return np.array(self.frames, dtype=np.float32)

    def render(self) -> np.ndarray:
        return self.env.render()

    def close(self):
        self.env.close()
