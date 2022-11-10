"""Define recurrent neural network"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(
            self, n_in, n_out, n_hid, device,
            alpha_time_scale=0.25, jij_std=0.045,
            activation='tanh',
            sigma_neu=0.05,
            use_bias=True,
            ffnn=False,
    ):
        super(RNN, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid, bias=use_bias)
        self.w_hh = nn.Linear(n_hid, n_hid, bias=use_bias)
        nn.init.uniform_(self.w_hh.weight, -jij_std, jij_std)
        self.w_out = nn.Linear(n_hid, n_out, bias=use_bias)

        self.activation = activation
        self.sigma_neu = sigma_neu

        self.device = device
        self.ffnn = ffnn

        self.alpha = torch.ones(self.n_hid) * alpha_time_scale
        self.alpha = self.alpha.to(self.device)

    def change_alpha(self, new_alpha_time_scale):
        self.alpha = torch.ones(self.n_hid) * new_alpha_time_scale
        self.alpha = self.alpha.to(self.device)

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            if self.activation == 'tanh':
                if self.ffnn:
                    tmp_hidden = self.w_in(input_signal[t])
                    hidden = torch.tanh(tmp_hidden)
                else:
                    activated = torch.tanh(hidden)
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
            elif self.activation == 'relu':
                if self.ffnn:
                    tmp_hidden = self.w_in(input_signal[t])
                    hidden = torch.nn.functional.relu(tmp_hidden)
                else:
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden)
                    tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
            else:
                raise ValueError

            output = self.w_out(hidden)
            output = torch.clamp(output, min=-20, max=20)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden


class RNNEI(RNN):
    def __init__(
        self, n_in, n_out, n_hid, device,
        alpha_time_scale=0.25, jij_std=0.045,
        activation='tanh',
        sigma_neu=0.05,
        use_bias=True,
        ffnn=False,
    ):
        super().__init__(
            n_in, n_out, n_hid, device,
            alpha_time_scale, jij_std,
            activation, sigma_neu,
            use_bias, ffnn,
        )
        self.w_hh.weight.data = torch.rand(n_hid, n_hid) / n_hid
        self.e_i_neuron = torch.eye(n_hid) * torch.from_numpy(np.array([1 if i < 240 else -1 for i in range(300)])).float()
        self.e_i_neuron = self.e_i_neuron.to(device)

    def forward(self, input_signal, hidden, length):
        w_rec = torch.mm(self.w_hh.weight, self.e_i_neuron)
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            if self.activation == 'tanh':
                if self.ffnn:
                    tmp_hidden = self.w_in(input_signal[t])
                    hidden = torch.tanh(tmp_hidden)
                else:
                    activated = torch.tanh(hidden)
                    tmp_hidden = self.w_in(input_signal[t]) + F.linear(activated, w_rec)
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
            elif self.activation == 'relu':
                if self.ffnn:
                    tmp_hidden = self.w_in(input_signal[t])
                    hidden = torch.nn.functional.relu(tmp_hidden)
                else:
                    tmp_hidden = self.w_in(input_signal[t]) + F.linear(hidden, w_rec)
                    tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
            else:
                raise ValueError

            output = self.w_out(hidden)
            output = torch.clamp(output, min=-20, max=20)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden
