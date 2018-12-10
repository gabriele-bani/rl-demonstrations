from torch import nn
import torch
import copy
from functools import partial

class PolynomialNetwork(nn.Module):

    def __init__(self, num_outputs=2, poly_order=3):
        nn.Module.__init__(self)
        
        self.input_len = 0
        self.poly_order = poly_order
        for x_power in range(poly_order + 1):
            for y_power in range(poly_order + 1 - x_power):
                self.input_len += 1

        self.l1 = nn.Linear(self.input_len, num_outputs)

    def polynomial(self, x, x_power, y_power):
        if len(x.shape) > 1:
            return x[:, 0] ** x_power * x[:, 1] ** y_power
        else:
            return x[0] ** x_power * x[1] ** y_power

    def forward(self, x):
        x_temp = []
        for x_power in range(self.poly_order + 1):
            for y_power in range(self.poly_order + 1 - x_power):
                x_temp.append(self.polynomial(x, x_power, y_power))
        if len(x.shape) > 1:
            x_temp = torch.stack(x_temp, 1)
        x = torch.FloatTensor(x_temp)
        out = self.l1(x)
        return out
