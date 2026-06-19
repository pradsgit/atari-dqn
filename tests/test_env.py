import numpy as np
import pytest
from env import AtariEnv


@pytest.fixture
def env():
    e = AtariEnv("Breakout")
    yield e
    e.close()


def test_reset_state_shape(env):
    state = env.reset()
    assert state.shape == (4, 84, 84)


def test_reset_state_dtype(env):
    state = env.reset()
    assert state.dtype == np.float32


def test_reset_state_value_range(env):
    state = env.reset()
    assert state.min() >= 0.0
    assert state.max() <= 1.0


def test_step_state_shape(env):
    env.reset()
    state, _, _, _ = env.step(0)
    assert state.shape == (4, 84, 84)


def test_step_reward_clipped(env):
    env.reset()
    # run several steps and check reward is always in {-1, 0, 1}
    for _ in range(10):
        _, reward, done, _ = env.step(env.action_space - 1)
        assert reward in (-1.0, 0.0, 1.0)
        if done:
            env.reset()


def test_step_done_is_bool(env):
    env.reset()
    _, _, done, _ = env.step(0)
    assert isinstance(done, (bool, np.bool_))


def test_prev_frame_cleared_on_reset(env):
    env.reset()
    env.step(0)
    assert env.prev_frame is not None
    env.reset()
    # prev_frame should be set again after first preprocess call in reset
    assert env.prev_frame is not None


def test_frame_stack_fills_on_reset(env):
    state = env.reset()
    # all 4 frames should be identical on reset (same first frame repeated)
    for i in range(1, 4):
        np.testing.assert_array_equal(state[0], state[i])


def test_frame_stack_updates_on_step(env):
    env.reset()
    # take enough steps for the game to produce a non-static frame
    state_prev = None
    for _ in range(20):
        state, _, done, _ = env.step(1)  # action 1 = FIRE, starts the game
        if done:
            env.reset()
        state_prev = state
    # after the game starts, frames should not all be identical
    assert not np.all(state_prev[0] == state_prev[3])


def test_action_space(env):
    assert isinstance(env.action_space, (int, np.integer))
    assert env.action_space > 0


def test_observation_space(env):
    assert env.observation_space == (4, 84, 84)


def test_no_reward_clipping():
    env = AtariEnv("Breakout", clip_rewards=False)
    env.reset()
    # just check it runs without error and reward is a number
    _, reward, _, _ = env.step(0)
    assert isinstance(reward, (int, float, np.floating))
    env.close()
