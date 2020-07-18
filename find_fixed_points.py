"""finding fixed points"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import yaml

sys.path.append('../')

from torch.autograd import Variable

import torch.utils.data as data
from fixed_point_analyzer import FixedPoint
from model import RecurrentNeuralNetwork


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
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior

    def __len__(self):
        return 10

    def __getitem__(self, item):
        # input signal
        signal_input = np.zeros(self.time_length + 1)
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

        return signal_input, target, signal_mu, signal_sigma, mu_posterior


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('fixed_points', exist_ok=True)
    os.makedirs('fixed_points/posterior', exist_ok=True)
    save_path = f'fixed_points/posterior/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25

    # model load
    model = RecurrentNeuralNetwork(n_in=1, n_out=1, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'], beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    model_path = f'trained_model/posterior/{model_name}/epoch_{cfg["TRAIN"]["NUM_EPOCH"]}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    eval_dataset = Posterior(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                             time_scale=cfg['MODEL']['ALPHA'],
                             mu_min=cfg['DATALOADER']['MU_MIN'],
                             mu_max=cfg['DATALOADER']['MU_MAX'],
                             sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                             sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                             mean_signal_length=cfg['DATALOADER']['MEAN_SIGNAL_LENGTH'],
                             variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                             mu_prior=cfg['MODEL']['MU_PRIOR'],
                             sigma_prior=cfg['MODEL']['SIGMA_PRIOR'])

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1,
                                                  num_workers=2, shuffle=True,
                                                  worker_init_fn=lambda x: np.random.seed())

    analyzer = FixedPoint(model=model, device=device, alpha=cfg['MODEL']['ALPHA'],
                          max_epochs=140000)
    count = 0
    for trial in range(5):
        for i, data in enumerate(eval_dataloader):
            inputs, target, signal_mu, signal_sigma, mu_posterior = data
            print(f'signal_mu: {signal_mu[0].item():.3f}')
            print(f'signal_sigma: {signal_sigma[0].item():.3f}')
            print(f'mu_posterior: {mu_posterior[0].item():.3f}')
            inputs, target = inputs.float(), target.long()
            print(inputs.shape)
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)

            hidden = torch.zeros(1, cfg['MODEL']['SIZE'])
            hidden = hidden.to(device)

            hidden = hidden.detach()
            hidden_list, output, hidden, _ = model(inputs, hidden)

            fixed_point, result_ok = analyzer.find_fixed_point(hidden_list[0, cfg['DATALOADER']['MEAN_SIGNAL_LENGTH']],
                                                               view=True)

            fixed_point = fixed_point.detach().cpu().numpy()

            print(fixed_point)
            fixed_point_tensor = torch.from_numpy(fixed_point).float()
            jacobian = analyzer.calc_jacobian(fixed_point_tensor, cfg['MODEL']['ACTIVATION'])

            print(f'output: {np.dot(model.w_out.weight.detach().cpu().numpy(), fixed_point)}')

            w, v = np.linalg.eig(jacobian)
            print('max eigenvalue', np.max(w.real))

            np.savetxt(os.path.join(save_path, f'fixed_point_{count:04d}.txt'), fixed_point)
            count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN fixed point analysis')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
