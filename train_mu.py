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

from mu_dataset import Mu
from model import RecurrentNeuralNetwork


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/mu', exist_ok=True)
    save_path = f'trained_model/mu/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25
    if 'VARIABLE_TIME_LENGTH' not in cfg['DATALOADER'].keys():
        cfg['DATALOADER']['VARIABLE_TIME_LENGTH'] = 0
    if 'FIXATION' not in cfg['DATALOADER'].keys():
        cfg['DATALOADER']['FIXATION'] = 1

    model = RecurrentNeuralNetwork(n_in=1, n_out=1, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'], beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    train_dataset = Mu(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                       time_scale=cfg['MODEL']['ALPHA'],
                       mu_min=cfg['DATALOADER']['MU_MIN'],
                       mu_max=cfg['DATALOADER']['MU_MAX'],
                       sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                       sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                       mean_signal_length=cfg['DATALOADER']['MEAN_SIGNAL_LENGTH'],
                       variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                       mu_prior=cfg['MODEL']['MU_PRIOR'],
                       sigma_prior=cfg['MODEL']['SIGMA_PRIOR'],
                       variable_time_length=cfg['DATALOADER']['VARIABLE_TIME_LENGTH'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)
    print('Epoch Loss Acc')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])

    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, target = data
            inputs, target = inputs.float(), target.float()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)

            # hidden_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            hidden_np = np.zeros((cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()
            variable_length = np.random.randint(-cfg['DATALOADER']['VARIABLE_TIME_LENGTH'],
                                                cfg['DATALOADER']['VARIABLE_TIME_LENGTH'] + 1)
            time_length = cfg['DATALOADER']['TIME_LENGTH'] + variable_length
            hidden_list, output, hidden = model(inputs, hidden, time_length)

            loss = torch.nn.MSELoss()(output[:, -1], target[:, :])
            for j in range(2, cfg['DATALOADER']['FIXATION'] + 1):
                loss += torch.nn.MSELoss()(output[:, -j], target[:, :])
            dummy_zero = torch.zeros([cfg['TRAIN']['BATCHSIZE'],
                                      time_length,
                                      cfg['MODEL']['SIZE']]).float().to(device)
            active_norm = torch.nn.MSELoss()(hidden_list, dummy_zero)

            loss += cfg['TRAIN']['ACTIVATION_LAMBDA'] * active_norm
            loss.backward()
            optimizer.step()

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            print(f'Train Epoch: {epoch}, Loss: {loss.item():.4f}, Norm term: {active_norm.item():.4f}')
            print('output', output[:5, -1, 0].cpu().detach().numpy())
            print('target', target[:5, -1].cpu().detach().numpy())

        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
