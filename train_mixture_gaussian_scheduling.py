"""training models"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.append('../')

from torch.autograd import Variable

from mixture_gaussian_dataset import MixtureGaussian
from model import RecurrentNeuralNetwork


def autocorrelation(data, k, device):
    """Returns the autocorrelation of the *k*th lag in a time series data.

    Parameters
    ----------
    data : one dimensional numpy array
    k : the *k*th lag in the time series data (indexing starts at 0)
    """

    # yの平均
    y_avg = torch.mean(data, dim=1).to(device)

    # 分子の計算
    sum_of_covariance = torch.zeros(y_avg.shape[0]).to(device)
    for i in range(k + 1, data.shape[1]):
        covariance = (data[:, i] - y_avg) * (data[:, i - (k + 1)] - y_avg)
        # print(covariance)
        sum_of_covariance += covariance[:, 0]

    # 分母の計算
    sum_of_denominator = torch.zeros(y_avg.shape[0]).to(device)
    for u in range(data.shape[1]):
        denominator = (data[:, u] - y_avg) ** 2
        sum_of_denominator += denominator[:, 0]

    return sum_of_covariance / sum_of_denominator


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    save_path = f'trained_model/mixture_gaussian_scheduling/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    eps_tensor = torch.tensor(0.00001).to(device)

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
    if 'NOISE_FIRST' not in cfg['MODEL']:
        cfg['MODEL']['NOISE_FIRST'] = False

    pre_sigma = cfg['DATALOADER']['PRE_SIGMA']

    model = RecurrentNeuralNetwork(
        n_in=cfg['DATALOADER']['INPUT_NEURON'],
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

    train_dataset = MixtureGaussian(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        time_scale=cfg['MODEL']['ALPHA'],
        mu_min=cfg['DATALOADER']['MU_MIN'],
        mu_max=cfg['DATALOADER']['MU_MAX'],
        input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        pre_sigma=pre_sigma,
    )

    valid_dataset = MixtureGaussian(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        time_scale=cfg['MODEL']['ALPHA'],
        mu_min=cfg['DATALOADER']['MU_MIN'],
        mu_max=cfg['DATALOADER']['MU_MAX'],
        input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        pre_sigma=pre_sigma,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
        num_workers=2, shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
        num_workers=2, shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    print(model)
    # print('Epoch Loss Acc')

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
    )
    a_list = torch.linspace(-2, 2, 40) + 0.05
    a_list = a_list.to(device)
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, target = data
            inputs, target = inputs.float(), target.float()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)

            if cfg['TRAIN']['RANDOM_START']:
                hidden_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            else:
                hidden_np = np.zeros((cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()

            hidden_list, output_list, hidden = model(inputs, hidden, cfg['DATALOADER']['TIME_LENGTH'])

            kldiv_loss = 0
            """
            for sample_id in range(cfg['TRAIN']['BATCHSIZE']):
                q_tensor_soft = torch.zeros(40).to(device)
                for j in range(30, cfg['DATALOADER']['TIME_LENGTH']):
                    q_tensor_soft += - torch.nn.Tanh()(
                        20 * ((output_list[sample_id, j] - a_list) ** 2 - 0.025)) / 2 + 0.5
                q_tensor_soft /= (cfg['DATALOADER']['TIME_LENGTH'] - 30)
                p_tensor = target[sample_id, 0]
                for j in range(40):
                    kldiv_loss += q_tensor_soft[j] * (q_tensor_soft[j] / (p_tensor[j] + eps_tensor) + eps_tensor).log()
            """
            q_tensor_soft = torch.zeros((cfg['TRAIN']['BATCHSIZE'], 40)).to(device)
            for j in range(30, cfg['DATALOADER']['TIME_LENGTH']):
                q_tensor_soft += - torch.nn.Tanh()(20 * ((output_list[:, j] - a_list) ** 2 - 0.025)) / 2 + 0.5
            q_tensor_soft /= (cfg['DATALOADER']['TIME_LENGTH'] - 30)
            p_tensor = target[:, 0]
            for j in range(40):
                _kldiv = q_tensor_soft[:, j] * (q_tensor_soft[:, j] / (p_tensor[:, j] + eps_tensor) + eps_tensor).log()
                kldiv_loss += torch.sum(_kldiv)

            autocorr_loss = 0
            for k in range(cfg['DATALOADER']['TIME_LENGTH'] - 30):
                # print(torch.abs(autocorrelation(output_list[:, 30:], k)))
                autocorr_loss += torch.abs(torch.sum(autocorrelation(output_list[:, 30:], k, device)))

            loss = kldiv_loss + 0.5 * autocorr_loss
            loss.backward()
            optimizer.step()
            # print(f'{i}, Epoch, {epoch}, KLDivLoss, {kldiv_loss.item():.3f}, AutoCorrLoss, {autocorr_loss.item():.3f}')

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            model.eval()
            for i, data in enumerate(valid_dataloader):
                inputs, target = data
                inputs, target = inputs.float(), target.float()
                inputs, target = Variable(inputs).to(device), Variable(target).to(device)

                if cfg['TRAIN']['RANDOM_START']:
                    hidden_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
                else:
                    hidden_np = np.zeros((cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
                hidden = torch.from_numpy(hidden_np).float()
                hidden = hidden.to(device)

                hidden_list, output_list, hidden = model(inputs, hidden, cfg['DATALOADER']['TIME_LENGTH'])

                kldiv_loss = 0
                for sample_id in range(cfg['TRAIN']['BATCHSIZE']):
                    q_tensor_soft = torch.zeros(40).to(device)
                    for j in range(30, cfg['DATALOADER']['TIME_LENGTH']):
                        q_tensor_soft += - torch.nn.Tanh()(
                            20 * ((output_list[sample_id, j] - a_list) ** 2 - 0.025)) / 2 + 0.5
                    q_tensor_soft /= (cfg['DATALOADER']['TIME_LENGTH'] - 30)
                    p_tensor = target[sample_id, 0]
                    for j in range(40):
                        kldiv_loss += q_tensor_soft[j] * (
                                q_tensor_soft[j] / (p_tensor[j] + eps_tensor) + eps_tensor).log()
                autocorr_loss = 0
                for k in range(cfg['DATALOADER']['TIME_LENGTH'] - 30):
                    # print(torch.abs(autocorrelation(output_list[:, 30:], k, device)))
                    autocorr_loss += torch.abs(torch.sum(autocorrelation(output_list[:, 30:], k, device)))

            print(f'Train Epoch, {epoch}, KLDivLoss, {kldiv_loss.item():.3f}, AutoCorrLoss, {autocorr_loss.item():.3f}')
            if kldiv_loss.item() < 20 and pre_sigma >= 0.4:
                pre_sigma -= 0.1
                print(pre_sigma)
                train_dataset = MixtureGaussian(
                    time_length=cfg['DATALOADER']['TIME_LENGTH'],
                    time_scale=cfg['MODEL']['ALPHA'],
                    mu_min=cfg['DATALOADER']['MU_MIN'],
                    mu_max=cfg['DATALOADER']['MU_MAX'],
                    input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
                    uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
                    pre_sigma=pre_sigma,
                )

                valid_dataset = MixtureGaussian(
                    time_length=cfg['DATALOADER']['TIME_LENGTH'],
                    time_scale=cfg['MODEL']['ALPHA'],
                    mu_min=cfg['DATALOADER']['MU_MIN'],
                    mu_max=cfg['DATALOADER']['MU_MAX'],
                    input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
                    uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
                    pre_sigma=pre_sigma,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                    num_workers=2, shuffle=True,
                    worker_init_fn=lambda x: np.random.seed(),
                )
                valid_dataloader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                    num_workers=2, shuffle=True,
                    worker_init_fn=lambda x: np.random.seed(),
                )

        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
