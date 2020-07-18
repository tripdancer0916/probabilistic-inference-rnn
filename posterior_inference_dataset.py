"""generating input and target"""

import numpy as np
import torch.utils.data as data


class Posterior(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            mu_min,
            mu_max,
            sigma_min,
            sigma_max,
            mean_signal_length,
            variable_signal_length,
            variable_time_length,
            mu_prior,
            sigma_prior):
        self.time_length = time_length
        self.time_scale = time_scale
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mean_signal_length = mean_signal_length
        self.variable_signal_length = variable_signal_length
        self.variable_time_length = variable_time_length
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal_input = np.zeros(self.time_length + self.variable_time_length + 1)
        v = np.random.randint(-self.variable_signal_length, self.variable_signal_length + 1)
        signal_mu = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        signal_sigma = np.random.rand() * (self.sigma_max - self.sigma_min) + self.sigma_min
        signal_length = self.mean_signal_length + v

        # signal
        signal = np.random.normal(signal_mu, signal_sigma, signal_length)
        signal_input[: signal_length] = signal

        # target
        mu_posterior = ((signal_sigma ** 2) * self.mu_prior +
                        (self.sigma_prior ** 2) * signal_mu) / (self.sigma_prior ** 2 + signal_sigma ** 2)
        target = np.array(mu_posterior)

        signal_input = np.expand_dims(signal_input, axis=1)
        target = np.expand_dims(target, axis=0)

        return signal_input, target
