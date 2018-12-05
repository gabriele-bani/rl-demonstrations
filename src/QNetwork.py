from torch import nn
import torch

class QNetwork(nn.Module):

    def __init__(self, num_inputs=2, num_hidden=128, num_outputs=2):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(num_inputs, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        out = self.l2(x)

        return out


