import torch
import pytest
from model import DQNModel


@pytest.fixture
def model():
    return DQNModel(in_channels=4, n_actions=6)


def test_output_shape(model):
    x = torch.zeros(1, 4, 84, 84)
    out = model(x)
    assert out.shape == (1, 6)


def test_output_shape_batch(model):
    x = torch.zeros(32, 4, 84, 84)
    out = model(x)
    assert out.shape == (32, 6)


def test_output_has_correct_n_actions():
    model = DQNModel(in_channels=4, n_actions=18)
    x = torch.zeros(1, 4, 84, 84)
    out = model(x)
    assert out.shape == (1, 18)


def test_output_can_be_negative(model):
    # Q-values can be negative — no ReLU on output layer
    x = torch.randn(32, 4, 84, 84)
    out = model(x)
    assert out.min().item() < 0


def test_conv_output_size(model):
    # conv block should output (batch, 64, 7, 7)
    x = torch.zeros(1, 4, 84, 84)
    out = model.conv(x)
    assert out.shape == (1, 64, 7, 7)


def test_forward_is_differentiable(model):
    x = torch.zeros(1, 4, 84, 84)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None


def test_different_inputs_give_different_outputs(model):
    x1 = torch.zeros(1, 4, 84, 84)
    x2 = torch.ones(1, 4, 84, 84)
    out1 = model(x1)
    out2 = model(x2)
    assert not torch.equal(out1, out2)
