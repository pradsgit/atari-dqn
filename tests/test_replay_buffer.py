import numpy as np
import pytest
from replay_buffer import ReplayBuffer


def make_transition(state_val=0.0):
    s = np.zeros((4, 84, 84), dtype=np.float32) + state_val
    a = 1
    r = 1.0
    s_next = np.zeros((4, 84, 84), dtype=np.float32) + state_val + 1
    done = False
    return s, a, r, s_next, done


@pytest.fixture
def buffer():
    return ReplayBuffer(max_size=100, batch_size=4)


def test_push_increases_length(buffer):
    assert len(buffer) == 0
    buffer.push(*make_transition())
    assert len(buffer) == 1


def test_push_multiple(buffer):
    for i in range(10):
        buffer.push(*make_transition(i))
    assert len(buffer) == 10


def test_buffer_overwrites_when_full():
    buffer = ReplayBuffer(max_size=5, batch_size=2)
    for i in range(7):
        buffer.push(*make_transition(i))
    assert len(buffer) == 5  # capped at max_size


def test_sample_raises_when_insufficient():
    buffer = ReplayBuffer(max_size=100, batch_size=32)
    for i in range(10):
        buffer.push(*make_transition(i))
    with pytest.raises(AssertionError):
        buffer.sample()


def test_sample_returns_correct_batch_size(buffer):
    for i in range(20):
        buffer.push(*make_transition(i))
    s, a, r, s_next, done = buffer.sample()
    assert len(s) == 4
    assert len(a) == 4
    assert len(r) == 4
    assert len(s_next) == 4
    assert len(done) == 4


def test_sample_state_shape(buffer):
    for i in range(20):
        buffer.push(*make_transition(i))
    s, _, _, s_next, _ = buffer.sample()
    assert s.shape == (4, 4, 84, 84)
    assert s_next.shape == (4, 4, 84, 84)


def test_sample_action_shape(buffer):
    for i in range(20):
        buffer.push(*make_transition(i))
    _, a, _, _, _ = buffer.sample()
    assert a.shape == (4,)


def test_sample_reward_shape(buffer):
    for i in range(20):
        buffer.push(*make_transition(i))
    _, _, r, _, _ = buffer.sample()
    assert r.shape == (4,)


def test_sample_done_shape(buffer):
    for i in range(20):
        buffer.push(*make_transition(i))
    _, _, _, _, done = buffer.sample()
    assert done.shape == (4,)


def test_sample_is_random():
    buffer = ReplayBuffer(max_size=1000, batch_size=4)
    for i in range(100):
        buffer.push(*make_transition(i))
    s1, _, _, _, _ = buffer.sample()
    s2, _, _, _, _ = buffer.sample()
    # two samples from 100 transitions are very unlikely to be identical
    assert not np.array_equal(s1, s2)
