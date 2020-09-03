"""define recurrent neural networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out, n_hid, device,
                 alpha_time_scale=0.25, beta_time_scale=0.1, jij_std=0.045, activation='tanh', sigma_neu=0.05, sigma_syn=0.002,
                 use_bias=True, anti_hebbian=True):
        super(RecurrentNeuralNetwork, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid, bias=False)
        self.w_hh = nn.Linear(n_hid, n_hid, bias=use_bias)
        nn.init.uniform_(self.w_hh.weight, -jij_std, jij_std)
        self.w_out = nn.Linear(n_hid, n_out, bias=False)

        self.activation = activation
        self.sigma_neu = sigma_neu
        self.sigma_syn = sigma_syn
        self.device = device

        self.alpha = torch.ones(self.n_hid) * alpha_time_scale
        self.beta = torch.ones(self.n_hid) * beta_time_scale
        self.alpha = self.alpha.to(self.device)
        self.beta = self.beta.to(self.device)
        self.anti_hebbian = anti_hebbian

    def change_alpha(self, new_alpha_time_scale):
        self.alpha = torch.ones(self.n_hid) * new_alpha_time_scale
        self.alpha = self.alpha.to(self.device)

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def anti_hebbian_synaptic_plasticity(self, num_batch, firing_rate, synapse, beta):
        outer_product = torch.zeros([num_batch, self.n_hid, self.n_hid]).to(self.device)
        for i in range(num_batch):
            outer_product[i, :, :] = torch.eye(self.n_hid)
        for i in range(num_batch):
            outer_product[i, :, :] = -torch.ger(firing_rate[i], firing_rate[i])
        return outer_product + torch.randn_like(synapse).to(self.device) * self.sigma_syn * torch.sqrt(beta)

    def hebbian_synaptic_plasticity(self, num_batch, firing_rate, synapse, beta):
        outer_product = torch.zeros([self.num, self.n_hid, self.n_hid]).to(self.device)
        for i in range(num_batch):
            outer_product[i, :, :] = torch.ger(firing_rate[i], firing_rate[i])
        return outer_product + torch.randn_like(synapse).to(self.device) * self.sigma_syn * torch.sqrt(beta)

    def forward(self, input_signal, hidden, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        different_j = torch.zeros((num_batch, self.n_hid, self.n_hid)).to(self.device)
        additional_w = torch.zeros((num_batch, self.n_hid, self.n_hid)).to(self.device)
        new_j = self.w_hh.weight
        for t in range(length):
            if self.activation == 'tanh':
                activated = torch.tanh(hidden)
                if self.beta[0].item() == 0:  # Short-term synaptic plasticityを考えない場合
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
                else:
                    different_j_activity = torch.matmul(activated.unsqueeze(1), different_j).squeeze(1)
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated) + different_j_activity

                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

                    if self.anti_hebbian:
                        additional_w = self.anti_hebbian_synaptic_plasticity(num_batch, activated, additional_w,
                                                                             self.beta)
                    else:
                        additional_w = self.hebbian_synaptic_plasticity(num_batch, activated, additional_w, self.beta)
                    new_j = new_j + self.beta * additional_w
                    different_j = new_j - self.w_hh.weight
            elif self.activation == 'relu':
                tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden)
                tmp_hidden = F.relu(tmp_hidden)
                neural_noise = self.make_neural_noise(hidden, self.alpha)
                hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
            elif self.activation == 'identity':
                activated = hidden
                if self.beta[0].item() == 0:  # Short-term synaptic plasticityを考えない場合
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
                else:
                    different_j_activity = torch.matmul(activated.unsqueeze(1), different_j).squeeze(1)
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated) + different_j_activity

                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

                    if self.anti_hebbian:
                        additional_w = self.anti_hebbian_synaptic_plasticity(num_batch, activated, additional_w,
                                                                             self.beta)
                    else:
                        additional_w = self.hebbian_synaptic_plasticity(num_batch, activated, additional_w, self.beta)
                    new_j = new_j + self.beta * additional_w
                    different_j = new_j - self.w_hh.weight
            else:
                raise ValueError

            output = 20 * nn.Tanh()(self.w_out(hidden))
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden
