"""generating input and target"""

import numpy as np
import torch.utils.data as data
from scipy.stats import norm


class CueCombination(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            mu_min,
            mu_max,
            mean_signal_length,
            variable_signal_length,
            variable_time_length,
            condition,
            input_neuron):
        self.time_length = time_length
        self.time_scale = time_scale
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mean_signal_length = mean_signal_length
        self.variable_signal_length = variable_signal_length
        self.variable_time_length = variable_time_length
        self.condition = condition
        self.input_neuron = input_neuron

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal1_input = np.zeros((self.time_length, self.input_neuron))
        signal2_input = np.zeros((self.time_length, self.input_neuron))

        phi = np.linspace(self.mu_min, self.mu_max, self.input_neuron)
        sigma_sq = 5

        signal_mu = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        if self.condition == 'all_gains':
            g_1, g_2 = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25], size=2)
        else:
            g_1 = np.random.choice([0.25, 1.25], size=1)
            g_2 = g_1

        # signal
        signal1_base = g_1 * np.exp(-(signal_mu - phi) ** 2 / (2.0 * sigma_sq))
        for t in range(self.time_length):
            signal1_input[t] = np.random.poisson(signal1_base)
        signal2_base = g_2 * np.exp(-(signal_mu - phi) ** 2 / (2.0 * sigma_sq))
        for t in range(self.time_length):
            signal2_input[t] = np.random.poisson(signal2_base)

        # target
        sigma_1 = np.sqrt(1 / g_1)
        sigma_2 = np.sqrt(1 / g_2)
        mu_posterior = ((sigma_1 ** 2) * signal_mu +
                        (sigma_2 ** 2) * signal_mu) / (sigma_1 ** 2 + sigma_2 ** 2)
        g_3 = g_1 + g_2
        sigma_posterior = np.sqrt(1 / g_3)
        n = np.linspace(-20, 20, 40)
        p = []
        for i in range(len(n)):
            p.append(norm.pdf(x=n[i], loc=mu_posterior, scale=sigma_posterior))
        target = np.array(p)

        signal_input = np.concatenate((signal1_input, signal2_input), axis=1)
        signal_input = signal_input.T
        target = np.expand_dims(target, axis=0)

        return signal_input, target
