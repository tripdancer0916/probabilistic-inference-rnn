"""Generating input and target for cue combination"""

import numpy as np
import torch.utils.data as data


class CueCombination(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            mu_min,
            mu_max,
            condition,
            input_neuron,
            uncertainty,
            fix_input=False,
            same_mu=True,
            beta=95,
    ):
        self.time_length = time_length
        self.time_scale = time_scale
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.condition = condition
        self.input_neuron = input_neuron
        self.uncertainty = uncertainty
        self.fix_input = fix_input
        self.same_mu = same_mu
        self.beta = beta

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        signal1_input = np.zeros((self.time_length, self.input_neuron))
        signal2_input = np.zeros((self.time_length, self.input_neuron))

        phi = np.linspace(self.mu_min, self.mu_max, self.input_neuron)
        sigma_sq = 5

        signal_mu1 = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        if self.same_mu:
            signal_mu2 = signal_mu1
        else:
            signal_mu2 = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        if self.condition == 'all_gains':
            g_1, g_2 = np.random.rand(2) + 0.25
        else:
            g_1 = np.random.choice([0.25, 1.25], size=1)
            g_2 = g_1

        # signal
        signal1_base = g_1 * np.exp(-(signal_mu1 - phi) ** 2 / (2.0 * sigma_sq))
        signal2_base = g_2 * np.exp(-(signal_mu2 - phi) ** 2 / (2.0 * sigma_sq))
        if self.fix_input:
            signal1_input_tmp = np.random.poisson(signal1_base)
            for t in range(self.time_length):
                signal1_input[t] = signal1_input_tmp
            signal2_input_tmp = np.random.poisson(signal1_base)
            for t in range(self.time_length):
                signal2_input[t] = signal2_input_tmp
        else:
            for t in range(self.time_length):
                signal1_input[t] = np.random.poisson(signal1_base)
            for t in range(self.time_length):
                signal2_input[t] = np.random.poisson(signal2_base)

        # target
        sigma_1 = np.sqrt(1 / g_1) * self.uncertainty
        sigma_2 = np.sqrt(1 / g_2) * self.uncertainty
        mu_posterior = ((sigma_1 ** 2) * signal_mu2 +
                        (sigma_2 ** 2) * signal_mu1) / (sigma_1 ** 2 + sigma_2 ** 2)
        g_3 = g_1 + g_2
        sigma_posterior = np.sqrt(1 / g_3) * self.uncertainty
        target_sample = np.random.normal(mu_posterior, sigma_posterior, 1000)
        a_list = np.linspace(-20, 20, 40) + 0.05
        p_soft = np.zeros(40)
        for i in range(1000):
            p_soft += -np.tanh(self.beta * ((target_sample[i] - a_list) ** 2 - 0.05**2)) / 2 + 0.5

        p_soft /= 1000

        signal_input = np.concatenate((signal1_input, signal2_input), axis=1)
        target = np.expand_dims(p_soft, axis=0)

        return signal_input, target


class CueCombinationPoint(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            mu_min,
            mu_max,
            condition,
            input_neuron,
            uncertainty,
            fix_input=False,
            same_mu=True,
    ):
        self.time_length = time_length
        self.time_scale = time_scale
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.condition = condition
        self.input_neuron = input_neuron
        self.uncertainty = uncertainty
        self.fix_input = fix_input
        self.same_mu = same_mu

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        signal1_input = np.zeros((self.time_length, self.input_neuron))
        signal2_input = np.zeros((self.time_length, self.input_neuron))

        phi = np.linspace(self.mu_min, self.mu_max, self.input_neuron)
        sigma_sq = 5

        signal_mu1 = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        if self.same_mu:
            signal_mu2 = signal_mu1
        else:
            signal_mu2 = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        if self.condition == 'all_gains':
            g_1, g_2 = np.random.rand(2) + 0.25
        else:
            g_1 = np.random.choice([0.25, 1.25], size=1)
            g_2 = g_1

        # signal
        signal1_base = g_1 * np.exp(-(signal_mu1 - phi) ** 2 / (2.0 * sigma_sq))
        signal2_base = g_2 * np.exp(-(signal_mu2 - phi) ** 2 / (2.0 * sigma_sq))
        if self.fix_input:
            signal1_input_tmp = np.random.poisson(signal1_base)
            for t in range(self.time_length):
                signal1_input[t] = signal1_input_tmp
            signal2_input_tmp = np.random.poisson(signal1_base)
            for t in range(self.time_length):
                signal2_input[t] = signal2_input_tmp
        else:
            for t in range(self.time_length):
                signal1_input[t] = np.random.poisson(signal1_base)
            for t in range(self.time_length):
                signal2_input[t] = np.random.poisson(signal2_base)

        # target
        sigma_1 = np.sqrt(1 / g_1) * self.uncertainty
        sigma_2 = np.sqrt(1 / g_2) * self.uncertainty
        mu_posterior = ((sigma_1 ** 2) * signal_mu2 + (sigma_2 ** 2) * signal_mu1) / (sigma_1 ** 2 + sigma_2 ** 2)

        signal_input = np.concatenate((signal1_input, signal2_input), axis=1)
        target = np.array([mu_posterior]*50)

        return signal_input, target
