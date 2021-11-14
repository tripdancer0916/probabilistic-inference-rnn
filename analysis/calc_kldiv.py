import argparse
import os
import sys

sys.path.append('../')

import numpy as np
import torch
import yaml
from scipy.stats import norm
from scipy.stats import pearsonr

from model import RecurrentNeuralNetwork


def make_signal_cc_task(N, time_length, mu_min, mu_max, uncertainty):
    input_signals = np.zeros([N, time_length, 200])
    phi = np.linspace(mu_min, mu_max, 100)
    sigma_sq = 5
    mu_post_list = []
    sigma_post_list = []
    for i in range(N):
        signal1_input = np.zeros([time_length, 100])
        signal2_input = np.zeros([time_length, 100])
        mu = np.random.rand() * (mu_max - mu_min) + mu_min
        g_1, g_2 = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25], size=2)
        signal1_base = g_1 * np.exp(-(mu - phi) ** 2 / (2.0 * sigma_sq))
        for t in range(time_length):
            signal1_input[t] = np.random.poisson(signal1_base)

        signal2_base = g_2 * np.exp(-(mu - phi) ** 2 / (2.0 * sigma_sq))
        for t in range(time_length):
            signal2_input[t] = np.random.poisson(signal2_base)

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


def main(config_path, sample_num, model_epoch):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(1)
    device = torch.device('cpu')

    model_name = os.path.splitext(os.path.basename(config_path))[0]
    print('model_name: ', model_name, 'sigma_neu: ', cfg['MODEL']['SIGMA_NEU'])

    # cfg['MODEL']['SIGMA_NEU'] = 0

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25

    model = RecurrentNeuralNetwork(n_in=200, n_out=1, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'], beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    model_path = f'../trained_model/cue_combination_sampling/{model_name}/epoch_{model_epoch}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_signal, mu_post_list, sigma_post_list = make_signal_cc_task(sample_num,
                                                                      time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                                                      mu_min=cfg['DATALOADER']['MU_MIN'],
                                                                      mu_max=cfg['DATALOADER']['MU_MAX'],
                                                                      uncertainty=cfg['DATALOADER']['UNCERTAINTY'])

    hidden_np = np.zeros((sample_num, cfg['MODEL']['SIZE']))
    hidden = torch.from_numpy(hidden_np).float()
    hidden = hidden.to(device)
    inputs = torch.from_numpy(input_signal).float()
    inputs = inputs.to(device)
    hidden_list, outputs, _ = model(inputs, hidden, cfg['DATALOADER']['TIME_LENGTH'])
    neural_dynamics = hidden_list.cpu().detach().numpy()

    outputs_np = outputs.detach().numpy()[:, :, 0]
    a_list = np.linspace(-20, 20, 40) + 0.5
    kl_div = 0
    kl_div_soft = 0
    kl_div_tmp = 0
    eps = 1e-10
    n = np.linspace(-20, 20, 40)
    for i in range(sample_num):
        q = np.histogram(outputs_np[i], bins=40, range=(-20, 20))[0] / 60
        # print('q:', np.sum(q))
        q_soft = np.zeros(40)
        for j in range(cfg['DATALOADER']['TIME_LENGTH']):
            q_soft += -np.tanh(10 * ((outputs_np[i, j] - a_list) ** 2 - 0.25)) / 2 + 0.5
        q_soft /= cfg['DATALOADER']['TIME_LENGTH']
        # print('q_soft: ', np.sum(q_soft))

        p = []
        for j in range(len(n)):
            p.append(norm.pdf(x=n[j], loc=mu_post_list[i], scale=sigma_post_list[i]))
        # print('p:', np.sum(p))

        # if i == 0:
            # print(p)
            # print(q)
            # print(q_soft)

        kl_div += np.sum([qi * np.log(qi / (pi + eps) + eps) for qi, pi in zip(q, p)])
        kl_div_soft += np.sum([qi * np.log(qi / (pi + eps) + eps) for qi, pi in zip(q_soft, p)])
        kl_div_tmp += np.sum([qi * np.log(qi / (pi + eps) + eps) for qi, pi in zip(q_soft, q)])

    pearson_correlation1 = pearsonr(
        np.mean(np.sum((neural_dynamics > 0)[:, :, :], axis=2), axis=1),
        np.std(outputs.detach().numpy()[:, :, 0], axis=1),
    )

    pearson_correlation2 = pearsonr(
        sigma_post_list,
        np.std(outputs.detach().numpy()[:, :, 0], axis=1)
    )

    print(f'KL divergence: {kl_div / sample_num:.3f}')
    print(f'KL divergence(soft): {kl_div_soft / sample_num:.3f}')
    # print(f'KL divergence(tmp): {kl_div_tmp / sample_num:.3f}')
    print(f'Pearson correlation1: {pearson_correlation1[0]:.3f}')
    print(f'Pearson correlation2: {pearson_correlation2[0]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sample_num', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=1000)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sample_num, args.epoch)
