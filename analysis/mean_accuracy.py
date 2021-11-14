import argparse
import os
import sys

import numpy as np
import yaml

sys.path.append('../')
import torch

from model import RecurrentNeuralNetwork


def make_signal_cc_task(N, time_length, mu_min, mu_max, uncertainty, fix_input=False, same_mean=False):
    input_signals = np.zeros([N, time_length, 200])
    phi = np.linspace(-20, 20, 100)
    sigma_sq = 5
    mu_post_list = []
    sigma_post_list = []
    for i in range(N):
        signal1_input = np.zeros([time_length, 100])
        signal2_input = np.zeros([time_length, 100])
        mu = np.random.rand() * (mu_max - mu_min) + mu_min
        # g_1, g_2 = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25], size=2)
        g_1, g_2 = np.random.rand(2) + 0.25
        # g_2 = g_1
        signal1_base = g_1 * np.exp(-(mu - phi) ** 2 / (2.0 * sigma_sq))
        signal2_base = g_2 * np.exp(-(mu - phi) ** 2 / (2.0 * sigma_sq))
        for t in range(time_length):
            signal1_input[t] = np.random.poisson(signal1_base)
        for t in range(time_length):
            signal2_input[t] = np.random.poisson(signal2_base)
        if fix_input:
            tmp1 = np.random.poisson(signal1_base)
            tmp2 = np.random.poisson(signal2_base)
            for t in range(time_length):
                signal1_input[t] = tmp1
                signal2_input[t] = tmp2

        # target
        sigma_1 = np.sqrt(1 / g_1) * uncertainty
        sigma_2 = np.sqrt(1 / g_2) * uncertainty
        mu_posterior = ((sigma_1 ** 2) * mu +
                        (sigma_2 ** 2) * mu) / (sigma_1 ** 2 + sigma_2 ** 2)
        g_3 = g_1 + g_2
        sigma_posterior = np.sqrt(1 / g_3) * uncertainty

        input_signals[i] = np.concatenate((signal1_input, signal2_input), axis=1)

        mu_post_list.append(mu_posterior)
        sigma_post_list.append(sigma_posterior)

    return input_signals, mu_post_list, sigma_post_list


def main(config_path, condition, num_epoch):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]
    print('model_name: ', model_name)

    torch.manual_seed(1)
    device = torch.device('cpu')

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25

    cfg['DATALOADER']['TIME_LENGTH'] = 60

    model = RecurrentNeuralNetwork(
        n_in=200,
        n_out=1,
        n_hid=cfg['MODEL']['SIZE'],
        device=device,
        alpha_time_scale=cfg['MODEL']['ALPHA'],
        activation=cfg['MODEL']['ACTIVATION'],
        sigma_neu=cfg['MODEL']['SIGMA_NEU'],
        use_bias=cfg['MODEL']['USE_BIAS'],
        ffnn=cfg['MODEL']['FFNN'],
    ).to(device)

    model_path = f'../trained_model/{condition}/{model_name}/epoch_{num_epoch}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sample_num = 300

    input_signal, mu_post_list, sigma_post_list = make_signal_cc_task(
        sample_num,
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        mu_min=-10,
        mu_max=10,
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        fix_input=False,
    )

    # hidden_np = np.zeros((sample_num, cfg['MODEL']['SIZE']))
    hidden_np = np.random.normal(0, 0.5, size=(sample_num, cfg['MODEL']['SIZE']))
    hidden = torch.from_numpy(hidden_np).float()
    hidden = hidden.to(device)
    inputs = torch.from_numpy(input_signal).float()
    inputs = inputs.to(device)
    hidden_list, outputs, _ = model(inputs, hidden, cfg['DATALOADER']['TIME_LENGTH'])
    neural_dynamics = hidden_list.cpu().detach().numpy()

    print(np.linalg.norm(mu_post_list - np.mean(outputs.detach().numpy()[:, 30:, 0], axis=1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('condition', type=str)
    parser.add_argument('--num_epoch', type=int, default=1000)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.condition, args.num_epoch)
