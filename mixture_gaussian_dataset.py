"""generating input and target"""

import numpy as np
import torch.utils.data as data


class MixtureGaussian(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            mu_min,
            mu_max,
            input_neuron,
            uncertainty,
            pre_sigma,
            pre_mu_1=-1,
            pre_mu_2=1,
            g_scale=1,
            fix=False,
            beta=95,
    ):
        self.time_length = time_length
        self.time_scale = time_scale
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.input_neuron = input_neuron
        self.uncertainty = uncertainty
        self.pre_sigma = pre_sigma
        self.pre_mu_1 = pre_mu_1
        self.pre_mu_2 = pre_mu_2
        self.g_scale = g_scale
        self.fix = fix
        self.beta = beta

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # print(item)
        signal_input = np.zeros((self.time_length, self.input_neuron))

        phi = np.linspace(self.mu_min, self.mu_max, self.input_neuron)
        sigma_sq = 5

        signal_mu = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        g = np.random.rand() + 0.25
        # g *= self.g_scale

        # signal
        signal_base = g * self.g_scale * np.exp(-(signal_mu - phi) ** 2 / (2.0 * sigma_sq))
        if self.fix:
            for t in range(self.time_length):
                signal_input[t] = signal_base
        else:
            for t in range(self.time_length):
                signal_input[t] = np.random.poisson(signal_base)

        # target
        sigma = np.sqrt(1 / g) * self.uncertainty
        mu_post_1 = ((sigma ** 2) * self.pre_mu_1 +
                     (self.pre_sigma ** 2) * signal_mu) / (sigma ** 2 + self.pre_sigma ** 2)
        mu_post_2 = ((sigma ** 2) * self.pre_mu_2 +
                     (self.pre_sigma ** 2) * signal_mu) / (sigma ** 2 + self.pre_sigma ** 2)
        sigma_posterior = sigma * self.pre_sigma / np.sqrt(sigma ** 2 + self.pre_sigma ** 2)
        normalize_factor = np.exp(-(signal_mu - self.pre_mu_1) ** 2 / (2 * (self.pre_sigma ** 2 + sigma ** 2))) + \
                           np.exp(-(signal_mu - self.pre_mu_2) ** 2 / (2 * (self.pre_sigma ** 2 + sigma ** 2)))
        pi_1 = np.exp(-(signal_mu - self.pre_mu_1) ** 2 / (2 * (self.pre_sigma ** 2 + sigma ** 2))) / normalize_factor
        # pi_2 = np.exp(-(signal_mu - self.pre_mu_2) ** 2 / (2 * (self.pre_sigma ** 2 + sigma ** 2))) / normalize_factor
        target_sample = []
        for i in range(1000):
            if np.random.rand() < pi_1:
                target_sample.append(np.random.normal(mu_post_1, sigma_posterior))
            else:
                target_sample.append(np.random.normal(mu_post_2, sigma_posterior))
        a_list = np.linspace(-2, 2, 40) + 0.05
        p_soft = np.zeros(40)
        for i in range(1000):
            p_soft += -np.tanh(self.beta * ((target_sample[i] - a_list) ** 2 - 0.05**2)) / 2 + 0.5

        p_soft /= 1000

        target = np.expand_dims(p_soft, axis=0)

        return signal_input, target
