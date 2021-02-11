"""define recurrent neural networks"""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.autograd import Variable

from cue_combination_dataset import series_of_com


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
            nu=1,
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
        self.nu = nu

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        # input signal
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
            signal1_input_tmp = series_of_com(signal1_base, self.nu)
            for t in range(self.time_length):
                signal1_input[t] = signal1_input_tmp
            signal2_input_tmp = series_of_com(signal2_base, self.nu)
            for t in range(self.time_length):
                signal2_input[t] = signal2_input_tmp
        else:
            for t in range(self.time_length):
                if self.nu == 1:
                    signal1_input[t] = np.random.poisson(signal1_base)
                else:
                    signal1_input[t] = series_of_com(signal1_base, self.nu)
            for t in range(self.time_length):
                if self.nu == 1:
                    signal2_input[t] = np.random.poisson(signal2_base)
                else:
                    signal2_input[t] = series_of_com(signal2_base, self.nu)

        # target
        sigma_1 = np.sqrt(1 / g_1) * self.uncertainty
        sigma_2 = np.sqrt(1 / g_2) * self.uncertainty
        mu_posterior = ((sigma_1 ** 2) * signal_mu2 +
                        (sigma_2 ** 2) * signal_mu1) / (sigma_1 ** 2 + sigma_2 ** 2)
        g_3 = g_1 + g_2
        sigma_posterior = np.sqrt(1 / g_3) * self.uncertainty

        signal_input = np.concatenate((signal1_input, signal2_input), axis=1)

        return signal_input, mu_posterior, sigma_posterior


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out, n_hid, device,
                 alpha_time_scale=0.25, jij_std=0.045,
                 activation='tanh',
                 sigma_neu=0.05,
                 use_bias=True,
                 ffnn=False,
                 noise_first=False):
        super(RecurrentNeuralNetwork, self).__init__()
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
        self.noise_first = noise_first

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
                if self.noise_first:
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = hidden + neural_noise
                    activated = torch.tanh(hidden)
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden
                else:
                    activated = torch.tanh(hidden)
                    tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)
                    neural_noise = self.make_neural_noise(hidden, self.alpha)
                    hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            elif self.activation == 'relu':
                if self.ffnn:
                    tmp_hidden = self.w_in(input_signal[t])
                    hidden = F.relu(tmp_hidden)
                else:
                    if self.noise_first:
                        neural_noise = self.make_neural_noise(hidden, self.alpha)
                        hidden = hidden + neural_noise
                        tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden)
                        tmp_hidden = F.relu(tmp_hidden)
                        hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden
                    else:
                        # tmp_hidden = F.relu(self.w_in(input_signal[t])) + F.relu(self.w_hh(hidden))
                        tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden)
                        tmp_hidden = F.relu(tmp_hidden)
                        neural_noise = self.make_neural_noise(hidden, self.alpha)
                        hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            elif self.activation == 'identity':
                activated = hidden
                tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)
                neural_noise = self.make_neural_noise(hidden, self.alpha)
                hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            else:
                raise ValueError

            output = self.w_out(hidden)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/cue_combination_sampling_mu_sigma', exist_ok=True)
    save_path = f'trained_model/cue_combination_sampling_mu_sigma/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if 'ALPHA' not in cfg['MODEL']:
        cfg['MODEL']['ALPHA'] = 0.25
    if 'VARIABLE_TIME_LENGTH' not in cfg['DATALOADER']:
        cfg['DATALOADER']['VARIABLE_TIME_LENGTH'] = 0
    if 'FIXATION' not in cfg['DATALOADER']:
        cfg['DATALOADER']['FIXATION'] = 1
    if 'RANDOM_START' not in cfg['TRAIN']:
        cfg['TRAIN']['RANDOM_START'] = True
    if 'FFNN' not in cfg['MODEL']:
        cfg['MODEL']['FFNN'] = False
    if 'FIX_INPUT' not in cfg['DATALOADER']:
        cfg['DATALOADER']['FIX_INPUT'] = False
    if 'SAME_MU' not in cfg['DATALOADER']:
        cfg['DATALOADER']['SAME_MU'] = True
    if 'NOISE_FIRST' not in cfg['MODEL']:
        cfg['MODEL']['NOISE_FIRST'] = False
    if 'NU' not in cfg['DATALOADER']:
        cfg['DATALOADER']['NU'] = 1

    model = RecurrentNeuralNetwork(
        n_in=2 * cfg['DATALOADER']['INPUT_NEURON'],
        n_out=1,
        n_hid=cfg['MODEL']['SIZE'],
        device=device,
        alpha_time_scale=cfg['MODEL']['ALPHA'],
        activation=cfg['MODEL']['ACTIVATION'],
        sigma_neu=cfg['MODEL']['SIGMA_NEU'],
        use_bias=cfg['MODEL']['USE_BIAS'],
        ffnn=cfg['MODEL']['FFNN'],
        noise_first=cfg['MODEL']['NOISE_FIRST'],
    ).to(device)

    train_dataset = CueCombination(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        time_scale=cfg['MODEL']['ALPHA'],
        mu_min=cfg['DATALOADER']['MU_MIN'],
        mu_max=cfg['DATALOADER']['MU_MAX'],
        condition=cfg['DATALOADER']['CONDITION'],
        input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        fix_input=cfg['DATALOADER']['FIX_INPUT'],
        same_mu=cfg['DATALOADER']['SAME_MU'],
        nu=cfg['DATALOADER']['NU'],
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, mu_target_list, sigma_target_list = data
            inputs, mu_target_list, sigma_target_list = inputs.float(), mu_target_list.float(), sigma_target_list.float()
            inputs, mu_target_list = Variable(inputs).to(device), Variable(mu_target_list).to(device)
            sigma_target_list = Variable(sigma_target_list).to(device)

            if cfg['TRAIN']['RANDOM_START']:
                hidden_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            else:
                hidden_np = np.zeros((cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()

            hidden_list, output_list, hidden = model(inputs, hidden, cfg['DATALOADER']['TIME_LENGTH'])

            musigma_loss = 0

            for sample_id in range(cfg['TRAIN']['BATCHSIZE']):
                mu_output = 0
                sigma_output = 0

                for j in range(cfg['DATALOADER']['TIME_LENGTH']):
                    mu_output += output_list[sample_id, j, 0]

                mu_output /= cfg['DATALOADER']['TIME_LENGTH']

                for j in range(cfg['DATALOADER']['TIME_LENGTH']):
                    sigma_output += (output_list[sample_id, j, 0] - mu_output) ** 2

                sigma_output /= cfg['DATALOADER']['TIME_LENGTH']

                musigma_loss += (mu_output - mu_target_list[sample_id]) ** 2 + \
                                (sigma_output - sigma_target_list[sample_id]) ** 2

            musigma_loss.backward()
            optimizer.step()

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            print(f'Train Epoch, {epoch}, Loss, {musigma_loss.item():.4f}')

        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
