"""generating input and target"""

import numpy as np
import torch.utils.data as data


class Mu(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            freq_min,
            freq_max,
            sigma_min,
            sigma_max,
            mean_signal_length,
            variable_signal_length,
            variable_time_length,
            mu_prior,
            sigma_prior):
        self.time_length = time_length
        self.time_scale = time_scale
        self.freq_min = freq_min
        self.freq_max = freq_max
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
        signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + self.freq_min
        signal_sigma = np.random.rand() * (self.sigma_max - self.sigma_min) + self.sigma_min
        signal_length = self.mean_signal_length + v

        # signal
        t = np.arange(0, signal_length * self.time_scale, self.time_scale)
        if len(t) != signal_length:
            t = t[:-1]
        first_signal = np.sin(signal_freq * t + phase_shift) + \
                            np.random.normal(0, self.sigma_in, first_signal_length)
        phase_shift = np.random.rand() * np.pi
        signal = np.random.normal(signal_mu, signal_sigma, signal_length)
        signal_input[: signal_length] = signal

        # target
        target = np.array(signal_mu)

        signal_input = np.expand_dims(signal_input, axis=1)
        target = np.expand_dims(target, axis=0)

        return signal_input, target
