# Model architecture

"""
Model architecture. There are several possible ways of parameterizing Q using a
neural network. Because Q maps history–action pairs to scalar estimates of their
Q-value, the history and the action have been used as inputs to the neural network
by some previous approaches24,26
. The main drawback of this type of architecture
is that a separate forward pass is required to compute the Q-value of each action,
resulting in a cost that scales linearly with the number of actions. We instead use an
architecture in which there is a separate output unit for each possible action, and
only the state representation is an input to the neural network. The outputs correspond to the predicted Q-values of the individual actions for the input state. The
main advantage of this type of architecture is the ability to compute Q-values for all
possible actions in a given state with only a single forward pass through the network.
The exact architecture, shown schematically in Fig. 1, is as follows. The input to
the neural network consists of an 84x84x4 image produced by the preprocessing map w. The first hidden layer convolves 32 filters of 8x8 with stride 4 with the
input image and applies a rectifier nonlinearity31,32. The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
This is followed by a third convolutional layer that convolves 64 filters of 3x3 with
stride 1 followed by a rectifier. The final hidden layer is fully-connected and consists of 512 rectifier units. The output layer is a fully-connected linear layer with a
single output for each valid action. The number of valid actions varied between 4
and 18 on the games we considered.
"""

import torch.nn as nn
import torch



class DQNModel(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_conv_output_size(in_channels), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_output_size(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        return self.conv(dummy).view(1, -1).shape[1]


    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x


