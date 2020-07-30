"""generating input and target"""

import numpy as np
import torch.utils.data as data


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
            condition):
        self.time_length = time_length
        self.time_scale = time_scale
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mean_signal_length = mean_signal_length
        self.variable_signal_length = variable_signal_length
        self.variable_time_length = variable_time_length
        self.condition = condition

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal1_input = np.zeros(self.time_length + self.variable_time_length + 1)
        signal2_input = np.zeros(self.time_length + self.variable_time_length + 1)
        v = np.random.randint(-self.variable_signal_length, self.variable_signal_length + 1)
        signal1_mu = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        signal2_mu = np.random.rand() * (self.mu_max - self.mu_min) + self.mu_min
        if self.condition == 'all_gains':
            sigma_1, sigma_2 = np.random.choice([0.8944, 1, 1.155, 1.414, 2.0], size=2)
        else:
            sigma_1, sigma_2 = np.random.choice([(0.8944, 0.8944), (2.0, 2.0)], size=1)
        signal_length = self.mean_signal_length + v

        # signal
        signal1 = np.random.normal(signal1_mu, sigma_1, signal_length)
        signal1_input[: signal_length] = signal1
        signal2 = np.random.normal(signal2_mu, sigma_2, signal_length)
        signal2_input[: signal_length] = signal2

        # target
        mu_posterior = ((sigma_1 ** 2) * signal2_mu +
                        (sigma_2 ** 2) * signal1_mu) / (sigma_1 ** 2 + sigma_2 ** 2)
        target = np.array(mu_posterior)

        signal_input = np.vstack((signal1_input, signal2_input))
        # signal_input = np.expand_dims(signal_input, axis=1)
        target = np.expand_dims(target, axis=0)

        return signal_input, target
