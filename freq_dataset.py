"""generating input and target"""

import numpy as np
import torch.utils.data as data


class Freq(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            freq_min,
            freq_max,
            sigma,
            mean_signal_length,
            variable_signal_length,
            variable_time_length,
            phase_shift):
        self.time_length = time_length
        self.time_scale = time_scale
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sigma = sigma
        self.mean_signal_length = mean_signal_length
        self.variable_signal_length = variable_signal_length
        self.variable_time_length = variable_time_length
        self.phase_shift = phase_shift

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal_input = np.zeros(self.time_length + self.variable_time_length + 1)
        v = np.random.randint(-self.variable_signal_length, self.variable_signal_length + 1)
        signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + self.freq_min
        signal_length = self.mean_signal_length + v

        # signal
        t = np.arange(0, signal_length * self.time_scale, self.time_scale)
        if len(t) != signal_length:
            t = t[:-1]
        phase_shift = np.random.rand() * np.pi * self.phase_shift

        signal = np.sin(signal_freq * t + phase_shift) + np.random.normal(0, self.sigma, signal_length)
        signal_input[: signal_length] = signal

        # target
        target = np.array(signal_freq)

        signal_input = np.expand_dims(signal_input, axis=1)
        target = np.expand_dims(target, axis=0)

        return signal_input, target
