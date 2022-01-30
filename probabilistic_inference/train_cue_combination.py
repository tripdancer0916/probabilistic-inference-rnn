"""Train model for the cue combination task (the sampling from posterior)."""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.autograd import Variable

from probabilistic_inference.dataset.cue_combination import CueCombination
from probabilistic_inference.model import RNN


def train(config_path):
    # Setting hyper-parameters
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # The directory to save to
    save_path = f'../trained_model/cue_combination_sampling/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # Copy config file to saved directory.
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'device: {device}')

    eps_tensor = torch.tensor(0.00001).to(device)

    model = RNN(
        n_in=2 * cfg['DATALOADER']['INPUT_NEURON'],
        n_out=1,
        n_hid=cfg['MODEL']['SIZE'],
        device=device,
        alpha_time_scale=cfg['MODEL']['ALPHA'],
        activation=cfg['MODEL']['ACTIVATION'],
        sigma_neu=cfg['MODEL']['SIGMA_NEU'],
        use_bias=cfg['MODEL']['USE_BIAS'],
        ffnn=cfg['MODEL']['FFNN'],
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
    )

    valid_dataset = CueCombination(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        time_scale=cfg['MODEL']['ALPHA'],
        mu_min=cfg['DATALOADER']['MU_MIN'],
        mu_max=cfg['DATALOADER']['MU_MAX'],
        condition='all_gains',
        input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        fix_input=cfg['DATALOADER']['FIX_INPUT'],
        same_mu=cfg['DATALOADER']['SAME_MU'],
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

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
    )
    a_list = torch.linspace(-20, 20, 40) + 0.05
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
            q_tensor_soft = torch.zeros(40).to(device)
            for j in range(cfg['DATALOADER']['TIME_LENGTH']):
                proxy = cfg['DATALOADER']['BETA'] * ((output_list[:, j] - a_list) ** 2 - 0.05**2)
                q_tensor_soft += - torch.nn.Tanh()(proxy) / 2 + 0.5
            q_tensor_soft /= (cfg['DATALOADER']['TIME_LENGTH'])
            p_tensor = target[:, 0]
            for j in range(40):
                _kldiv = q_tensor_soft[:, j] * (q_tensor_soft[:, j] / (p_tensor[:, j] + eps_tensor) + eps_tensor).log()
                kldiv_loss += torch.sum(_kldiv)

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
                    for j in range(cfg['DATALOADER']['TIME_LENGTH']):
                        proxy = cfg['DATALOADER']['BETA'] * ((output_list[sample_id, j] - a_list) ** 2 - 0.05 ** 2)
                        q_tensor_soft += - torch.nn.Tanh()(proxy) / 2 + 0.5
                    q_tensor_soft /= (cfg['DATALOADER']['TIME_LENGTH'])
                    p_tensor = target[sample_id, 0]
                    for j in range(40):
                        proxy = q_tensor_soft[j] / (p_tensor[j] + eps_tensor) + eps_tensor
                        kldiv_loss += q_tensor_soft[j] * proxy.log()

            print(f'Train Epoch, {epoch}, Loss, {kldiv_loss.item():.4f}')

        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    train(args.config_path)
