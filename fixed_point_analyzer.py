"""class for fixed point analysis"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class FixedPoint(object):
    def __init__(self, model, device, alpha, gamma=0.01, speed_tor=1e-10, max_epochs=1600000,
                 lr_decay_epoch=1000):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.speed_tor = speed_tor
        self.max_epochs = max_epochs
        self.lr_decay_epoch = lr_decay_epoch
        self.alpha = torch.ones(model.n_hid) * alpha
        self.alpha = self.alpha.to(device)
        self.model.eval()

    def calc_speed(self, hidden_activated, activation):
        if activation == 'tanh':
            activated = torch.tanh(hidden_activated)
            tmp_hidden = self.model.w_hh(activated)
            tmp_hidden = (1 - self.alpha) * hidden_activated + self.alpha * tmp_hidden
        elif activation == 'relu':
            tmp_hidden = self.model.w_hh(hidden_activated)
            tmp_hidden = F.relu(tmp_hidden)
            tmp_hidden = (1 - self.alpha) * hidden_activated + self.alpha * tmp_hidden

        speed = torch.norm(tmp_hidden - hidden_activated)

        return speed

    def find_fixed_point(self, init_hidden, view=False):
        new_hidden = init_hidden.clone()
        gamma = self.gamma
        result_ok = True
        i = 0
        while True:
            hidden_activated = Variable(new_hidden).to(self.device)
            hidden_activated.requires_grad = True
            speed = self.calc_speed(hidden_activated, self.model.activation)
            if view and i % 5000 == 0:
                print(f'epoch: {i}, speed={speed.item()}')
            if speed.item() < self.speed_tor:
                print(f'epoch: {i}, speed={speed.item()}')
                break
            speed.backward()
            if i % self.lr_decay_epoch == 0 and 0 < i:
                gamma *= 0.9
            if i == self.max_epochs:
                print(f'forcibly finished. speed={speed.item()}')
                result_ok = False
                break
            i += 1

            new_hidden = hidden_activated - gamma * hidden_activated.grad

        fixed_point = new_hidden
        return fixed_point, result_ok

    def calc_jacobian(self, fixed_point, activation):
        if activation == 'tanh':
            tanh_dash = 1 - np.tanh(fixed_point) ** 2

            w_hh = self.model.w_hh.weight.data.numpy()
            jacobian = np.zeros((self.model.n_hid, self.model.n_hid))
            for i in range(self.model.n_hid):
                for j in range(self.model.n_hid):
                    jacobian[i, j] = tanh_dash[j] * w_hh[i, j]
                    if i == j:
                        jacobian[i, j] -= 1

        elif activation == 'relu':
            relu_dash = np.array([1 if x >= 0 else 0 for x in fixed_point])
            w_hh = self.model.w_hh.weight.data.numpy()
            jacobian = np.zeros((self.model.n_hid, self.model.n_hid))
            for i in range(self.model.n_hid):
                for j in range(self.model.n_hid):
                    jacobian[i, j] = relu_dash[j] * w_hh[i, j]
                    if i == j:
                        jacobian[i, j] -= 1

        return jacobian
