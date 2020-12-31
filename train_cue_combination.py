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

from cue_combination_dataset import CueCombination
from model import RecurrentNeuralNetwork


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/cue_combination_sampling', exist_ok=True)
    save_path = f'trained_model/cue_combination_sampling/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    eps_tensor = torch.tensor(0.00001).to(device)

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25
    if 'VARIABLE_TIME_LENGTH' not in cfg['DATALOADER'].keys():
        cfg['DATALOADER']['VARIABLE_TIME_LENGTH'] = 0
    if 'FIXATION' not in cfg['DATALOADER'].keys():
        cfg['DATALOADER']['FIXATION'] = 1
    if 'RANDOM_START' not in cfg['TRAIN'].keys():
        cfg['TRAIN']['RANDOM_START'] = True
    if 'FFNN' not in cfg['MODEL'].keys():
        cfg['MODEL']['FFNN'] = False
    if 'FIX_INPUT' not in cfg['DATALOADER'].keys():
        cfg['DATALOADER']['FIX_INPUT'] = False
    if 'SAME_MU' not in cfg['DATALOADER'].keys():
        cfg['DATALOADER']['SAME_MU'] = True
    if 'NOISE_FIRST' not in cfg['MODEL'].keys():
        cfg['MODEL']['NOISE_FIRST'] = False

    model = RecurrentNeuralNetwork(n_in=2 * cfg['DATALOADER']['INPUT_NEURON'], n_out=1, n_hid=cfg['MODEL']['SIZE'],
                                   device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   ffnn=cfg['MODEL']['FFNN'],
                                   noise_first=cfg['MODEL']['NOISE_FIRST']).to(device)

    train_dataset = CueCombination(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                   time_scale=cfg['MODEL']['ALPHA'],
                                   mu_min=cfg['DATALOADER']['MU_MIN'],
                                   mu_max=cfg['DATALOADER']['MU_MAX'],
                                   mean_signal_length=cfg['DATALOADER']['MEAN_SIGNAL_LENGTH'],
                                   variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                   variable_time_length=cfg['DATALOADER']['VARIABLE_TIME_LENGTH'],
                                   condition=cfg['DATALOADER']['CONDITION'],
                                   input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
                                   uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
                                   fix_input=cfg['DATALOADER']['FIX_INPUT'],
                                   same_mu=cfg['DATALOADER']['SAME_MU'])

    valid_dataset = CueCombination(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                   time_scale=cfg['MODEL']['ALPHA'],
                                   mu_min=cfg['DATALOADER']['MU_MIN'],
                                   mu_max=cfg['DATALOADER']['MU_MAX'],
                                   mean_signal_length=cfg['DATALOADER']['MEAN_SIGNAL_LENGTH'],
                                   variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                   variable_time_length=cfg['DATALOADER']['VARIABLE_TIME_LENGTH'],
                                   condition='all_gains',
                                   input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
                                   uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
                                   fix_input=cfg['DATALOADER']['FIX_INPUT'],
                                   same_mu=cfg['DATALOADER']['SAME_MU'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)
    # print('Epoch Loss Acc')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    a_list = torch.linspace(-20, 20, 40) + 0.5
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
            # print(target.shape)

            # print(output_list[0])
            for sample_id in range(cfg['TRAIN']['BATCHSIZE']):
                q_tensor_soft = torch.zeros(40).to(device)
                for j in range(cfg['DATALOADER']['TIME_LENGTH']):
                    q_tensor_soft += - torch.nn.Tanh()(2 * ((output_list[sample_id, j] - a_list) ** 2 - 0.25)) / 2 + 0.5
                    # print(q_tensor_soft)
                q_tensor_soft /= cfg['DATALOADER']['TIME_LENGTH']
                # print(q_tensor_soft)
                p_tensor = target[sample_id, 0]
                # print(p_tensor)
                # print(q_tensor_soft[0], p_tensor[0], (q_tensor_soft[0] / (p_tensor[0] + eps_tensor) + eps_tensor).log())
                # print(q_tensor_soft[0] * (q_tensor_soft[0] / p_tensor[0] + eps_tensor).log())
                for j in range(40):
                    kldiv_loss += q_tensor_soft[j] * (q_tensor_soft[j] / (p_tensor[j] + eps_tensor) + eps_tensor).log()
                    # print(kldiv_loss)

                # print(kldiv_loss)

            kldiv_loss.backward()
            optimizer.step()

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
                            2 * ((output_list[sample_id, j] - a_list) ** 2 - 0.25)) / 2 + 0.5
                    q_tensor_soft /= (cfg['DATALOADER']['TIME_LENGTH'] - 30)
                    p_tensor = target[sample_id, 0]
                    for j in range(40):
                        kldiv_loss += q_tensor_soft[j] * (
                                    q_tensor_soft[j] / (p_tensor[j] + eps_tensor) + eps_tensor).log()

            print(f'Train Epoch, {epoch}, Loss, {kldiv_loss.item():.4f}')
            # print('output mean', np.mean(output_list[:5, -5:, 0].cpu().detach().numpy(), axis=1))
            # print('output variability', np.std(output_list[:5, :, 0].cpu().detach().numpy(), axis=1))
            # print('target', np.argmax(target[:5, 0].cpu().detach().numpy(), axis=1) - 20)
            # print('target_uncertainty', 1 / np.max(target[:5, 0].cpu().detach().numpy(), axis=1))

        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
