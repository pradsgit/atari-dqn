import torch
import numpy as np
import pytest
from agent import DQNAgent


@pytest.fixture
def agent():
    return DQNAgent(in_channels=4, n_actions=6, lr=1e-4, gamma=0.99, device="cpu")


def make_batch(batch_size=32, n_actions=6):
    s      = np.random.rand(batch_size, 4, 84, 84).astype(np.float32)
    a      = np.random.randint(0, n_actions, size=(batch_size,))
    r      = np.random.rand(batch_size).astype(np.float32)
    s_next = np.random.rand(batch_size, 4, 84, 84).astype(np.float32)
    done   = np.zeros(batch_size, dtype=np.float32)
    return s, a, r, s_next, done


# --- select_action ---

def test_select_action_random_when_epsilon_1(agent):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    actions = {agent.select_action(state, epsilon=1.0) for _ in range(50)}
    assert len(actions) > 1  # should see multiple different actions


def test_select_action_greedy_when_epsilon_0(agent):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    actions = {agent.select_action(state, epsilon=0.0) for _ in range(10)}
    assert len(actions) == 1  # same state always gives same action


def test_select_action_valid_range(agent):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    for _ in range(20):
        action = agent.select_action(state, epsilon=0.5)
        assert 0 <= action < agent.n_actions


# --- train_step ---

def test_train_step_returns_loss(agent):
    loss = agent.train_step(*make_batch())
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_train_step_updates_online_weights(agent):
    params_before = [p.clone() for p in agent.online_net.parameters()]
    agent.train_step(*make_batch())
    params_after = list(agent.online_net.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    assert changed


def test_train_step_does_not_update_target_weights(agent):
    params_before = [p.clone() for p in agent.target_net.parameters()]
    agent.train_step(*make_batch())
    params_after = list(agent.target_net.parameters())
    unchanged = all(torch.equal(b, a) for b, a in zip(params_before, params_after))
    assert unchanged


def test_train_step_done_mask(agent):
    s, a, r, s_next, _ = make_batch()
    done_all = np.ones_like(r)  # all episodes ended
    loss = agent.train_step(s, a, r, s_next, done_all)
    assert isinstance(loss, float)


# --- sync_target ---

def test_sync_target_copies_weights(agent):
    # train to make online and target diverge
    for _ in range(5):
        agent.train_step(*make_batch())

    # verify they differ before sync
    differs = any(
        not torch.equal(op, tp)
        for op, tp in zip(agent.online_net.parameters(), agent.target_net.parameters())
    )
    assert differs

    # sync and verify they match
    agent.sync_target()
    matches = all(
        torch.equal(op, tp)
        for op, tp in zip(agent.online_net.parameters(), agent.target_net.parameters())
    )
    assert matches
