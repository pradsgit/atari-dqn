import torch
import torch.nn as nn
from model import DQNModel
import numpy as np
import random
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, in_channels: int, n_actions: int, lr: float, gamma: float, device: str):
        self.gamma = gamma
        self.device = device
        self.n_actions = n_actions

        self.online_net = DQNModel(in_channels, n_actions).to(device)
        self.target_net = DQNModel(in_channels, n_actions).to(device)

        # copy online weights to target and freeze
        self.target_net.load_state_dict(self.online_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr, weight_decay=1e-4)
    
    def select_action(self, state: np.ndarray, epsilon: float) -> int: 

        if random.random() < epsilon:
            # explore - take a random action
            return random.randint(0, self.n_actions-1)
        else:
            # exploit - take the best action
            with torch.no_grad():
                output = self.online_net(torch.tensor(state).unsqueeze(0).to(self.device))
            # return argmax of the output
            return torch.argmax(output).item()

    def train_step(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, s_next: np.ndarray, done: np.ndarray) -> float:
        """
        1. Take a minibatch from the replay buffer (s, a, r, s', done)
        2. Compute target: r + γ · max_a' target_net(s')
        3. Compute prediction: online_net(s, a)
        4. Compute loss: (prediction - target)²
        5. Backprop and update online network weights
        """
        s_tensor      = torch.tensor(s,      dtype=torch.float32).to(self.device)
        a_tensor      = torch.tensor(a,      dtype=torch.long).to(self.device)
        r_tensor      = torch.tensor(r,      dtype=torch.float32).to(self.device)
        s_next_tensor = torch.tensor(s_next, dtype=torch.float32).to(self.device)
        done_tensor   = torch.tensor(done,   dtype=torch.float32).to(self.device)

        # max Q-value at s' from frozen target network
        with torch.no_grad():
            max_q_next = torch.max(self.target_net(s_next_tensor), dim=1).values

        # target: r if done, else r + γ · max Q(s')
        target = r_tensor + self.gamma * max_q_next * (1 - done_tensor)

        # Q-value for the action actually taken
        preds = self.online_net(s_tensor).gather(1, a_tensor.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(preds, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def sync_target(self):
        """
        sync online net weights to target weights
        """
        self.target_net.load_state_dict(self.online_net.state_dict())
