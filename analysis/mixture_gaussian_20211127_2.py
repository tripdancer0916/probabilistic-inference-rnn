import os
import sys

import numpy as np
import yaml

sys.path.append('../')
import torch

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from matplotlib import cm

from model import RecurrentNeuralNetwork


def make_signal_cc_task(N, time_length, mu_min, mu_max, uncertainty=10, pre_sigma=3., specified_mu=None):
    pre_mu_1 = -1
    pre_mu_2 = 1
    input_signals = np.zeros([N, time_length, 100])
    phi = np.linspace(-1, 1, 100)
    sigma_sq = 5
    target_list = np.zeros([N, 40])

    for i in range(N):
        signal_input = np.zeros([time_length, 100])
        if specified_mu is None:
            mu = np.random.rand() * (mu_max - mu_min) + mu_min
        else:
            mu = specified_mu
        g = np.random.rand() + 0.25
        signal_base = g * np.exp(-(mu - phi) ** 2 / (2.0 * sigma_sq))
        for t in range(time_length):
            signal_input[t] = np.random.poisson(signal_base)

        # target
        sigma = np.sqrt(1 / g) * uncertainty
        mu_post_1 = ((sigma ** 2) * pre_mu_1 +
                     (pre_sigma ** 2) * mu) / (sigma ** 2 + pre_sigma ** 2)
        mu_post_2 = ((sigma ** 2) * pre_mu_2 +
                     (pre_sigma ** 2) * mu) / (sigma ** 2 + pre_sigma ** 2)
        sigma_posterior = sigma * pre_sigma / np.sqrt(sigma ** 2 + pre_sigma ** 2)
        normalize_factor = np.exp(-(mu - pre_mu_1) ** 2 / (2 * (pre_sigma ** 2 + sigma ** 2))) + \
                           np.exp(-(mu - pre_mu_2) ** 2 / (2 * (pre_sigma ** 2 + sigma ** 2)))
        pi_1 = np.exp(-(mu - pre_mu_1) ** 2 / (2 * (pre_sigma ** 2 + sigma ** 2))) / normalize_factor

        target_sample = []
        for j in range(1000):
            if np.random.rand() < pi_1:
                target_sample.append(np.random.normal(mu_post_1, sigma_posterior))
            else:
                target_sample.append(np.random.normal(mu_post_2, sigma_posterior))
        a_list = np.linspace(-2, 2, 40) + 0.05
        p_soft = np.zeros(40)
        for j in range(1000):
            p_soft += -np.tanh(2 * ((target_sample[j] - a_list) ** 2 - 0.025)) / 2 + 0.5

        p_soft /= 1000

        input_signals[i] = signal_input
        target_list[i] = p_soft

    return input_signals, target_list


config_path = '../cfg/mixture_gaussian/20211127_2.cfg'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

model_name = os.path.splitext(os.path.basename(config_path))[0]
print('model_name: ', model_name)

torch.manual_seed(1)
device = torch.device('cpu')

if 'ALPHA' not in cfg['MODEL'].keys():
    cfg['MODEL']['ALPHA'] = 0.25


model = RecurrentNeuralNetwork(n_in=100, n_out=1, n_hid=cfg['MODEL']['SIZE'], device=device,
                               alpha_time_scale=cfg['MODEL']['ALPHA'],
                               activation=cfg['MODEL']['ACTIVATION'],
                               # sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                               sigma_neu=0.05,
                               use_bias=cfg['MODEL']['USE_BIAS'],
                               ffnn=False).to(device)

model_path = f'../trained_model/mixture_gaussian_scheduling/{model_name}/epoch_380.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

sample_num = 1
input_signal, target_list = make_signal_cc_task(
    sample_num,
    time_length=5000,
    mu_min=-1,
    mu_max=1,
    uncertainty=1,
    pre_sigma=0.3,
    specified_mu=0,
)

hidden_np = np.zeros((sample_num, cfg['MODEL']['SIZE']))
hidden = torch.from_numpy(hidden_np).float()
hidden = hidden.to(device)
inputs = torch.from_numpy(input_signal).float()
inputs = inputs.to(device)
hidden_list, outputs, _ = model(inputs, hidden, 5000)
neural_dynamics = hidden_list.cpu().detach().numpy()

time_series = outputs.detach().numpy()[0, 30:, 0]

pca = PCA(n_components=3)
pca.fit(neural_dynamics[0, 30:, :])

fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)
ax.view_init(elev=45, azim=70)

# 軸ラベルの設定
ax.set_xlabel('PC1', fontsize=14)
ax.set_ylabel('PC2', fontsize=14)
ax.set_zlabel('PC3', fontsize=14)

trajectory = pca.transform(neural_dynamics[0, 30:])

internal_dynamics = ax.scatter(
    trajectory[:1000, 0],
    trajectory[:1000, 1],
    trajectory[:1000, 2],
    c=time_series[:1000],
    cmap=cm.jet, s=150, marker='.', lw=0, zorder=2,
)

cbar = fig.colorbar(internal_dynamics, shrink=0.75)
cbar.set_label(r'$y(t)$', fontsize=22)
plt.show()
