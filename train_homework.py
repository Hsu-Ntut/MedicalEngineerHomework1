from copy import deepcopy
from typing import Tuple
from icecream import ic
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wfdb
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import accelerate
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW, Optimizer
from torchaudio.transforms import Spectrogram
from torchvision.transforms import transforms as T
# from torchvision.models import resnet18
from torch.nn.functional import one_hot
from torchaudio.models import wav2vec2_model, wav2vec2_base, conv_tasnet_base
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, CyclicLR
from npo.models.homework_model import resnet18, resnet50
from npo.homework import utils as HU
from torchaudio.functional import add_noise
import functools
import tensorboardX

ALL_CATEGORIES = ['A', "L", "R"]
accelerator: accelerate.Accelerator = None


def get_loader(train_arg: dict, val_arg: dict, config, **kwargs) -> Tuple[DataLoader, DataLoader, int, np.ndarray]:
    final_nc = 23
    pack = HU.load_data(**config['utils_config'])
    x_train, y_train, x_test, y_test, _, weights = pack
    if not torch.is_tensor(weights):
        weights = torch.from_numpy(weights)
    print(y_train.shape)
    print(np.argwhere(y_train < 0))
    tds = TensorDataset(torch.from_numpy(x_train).float(),
                        one_hot(torch.from_numpy(y_train).long(), final_nc).float())
    vds = TensorDataset(torch.from_numpy(x_test).float(),
                        one_hot(torch.from_numpy(y_test).long(), final_nc).float())
    return DataLoader(tds, **train_arg), DataLoader(vds, **val_arg), final_nc, weights


def main(config):
    tloader, vloader, nc, weights = get_loader(
        train_arg={
            'batch_size': 512,
            'shuffle': True,
            'num_workers': 2,
            'pin_memory': True,
        },
        val_arg={
            'batch_size': 1024,
            'shuffle': False,
            'num_workers': 2,
            'pin_memory': True,
        }, config=config
    )
    model = resnet50(num_classes=nc, proj=False)
    optimizer: Optimizer = AdamW(model.parameters(), lr=.1, weight_decay=5e-4)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-9)
    # up: 26 epochs, down: 52
    scheduler = CyclicLR(optimizer, step_size_up=26, step_size_down=52, base_lr=1e-8, max_lr=.1, cycle_momentum=False)
    lfn = [nn.CrossEntropyLoss(weights), nn.CosineEmbeddingLoss()]
    model, optimizer, tloader, vloader, scheduler, lfn[0] = accelerator.prepare(model, optimizer,
                                                                             tloader, vloader,
                                                                             scheduler, lfn[0])
    lfn[0], lfn[1] = accelerator.prepare(*lfn)
    EPOCH = config['n_epoch']
    epoch_tot = len(tloader)
    step = 0
    loss_buf = float('inf')
    for epoch in range(EPOCH):
        tloss = train(model, tloader, optimizer, lfn, n_epoch=epoch)
        step += epoch_tot
        vloss = val(model, vloader, nn.CrossEntropyLoss(), step=step, n_epoch=epoch)

        scheduler.step()
        accelerator.log({
            'lr': scheduler.get_lr(),
            'Epoch Train Loss': tloss,
            'Epoch Val Loss': vloss,
            'step': step,
            'epoch': epoch
        }, step=step)

        if vloss < loss_buf:
            torch.save(accelerator.unwrap_model(model), f'{accelerator.project_dir}/best.pth')
            loss_buf = vloss
    torch.save(accelerator.unwrap_model(model), f'{accelerator.project_dir}/last.pth')


def train(model, tloader, optimizer, lfn, **kwargs):
    n_epoch = kwargs.get('n_epoch', 0)
    loader_tot = len(tloader)
    pbar = tqdm(enumerate(tloader), total=loader_tot, desc=f'Train:[{n_epoch}]')
    model.train()
    tloss = .0

    for batch_idx, (data, target) in pbar:
        optimizer.zero_grad(set_to_none=True)
        pred = model(data)
        loss = lfn[0](pred, target.squeeze(1))
        accelerator.backward(loss)
        optimizer.step()
        # Training iterative is done.

        tloss += loss.item()
        pbar.set_postfix({
            'iter loss': loss.item(),
            'train loss': tloss / (batch_idx + 1),
            'step': n_epoch * loader_tot + batch_idx + 1
        })
        accelerator.log({
            'train loss': tloss / (batch_idx + 1),
            'train iter loss': loss.item(),
            'step': n_epoch * loader_tot + batch_idx + 1,
            'lr': optimizer.param_groups[0]['lr']
        }, step=n_epoch * loader_tot + batch_idx + 1)
    optimizer.zero_grad(set_to_none=True)
    return tloss / (batch_idx + 1)

@torch.no_grad()
def val(model: torch.nn.Module, vloader: DataLoader, lfn, **kwargs):
    n_epoch = kwargs.get('n_epoch', 0)
    model.eval()
    vloss = .0
    loader_tot = len(vloader)
    pbar = tqdm(enumerate(vloader), total=loader_tot, desc='Validation')
    for batch_idx, (data, target) in pbar:
        pred= model(data)
        loss = lfn(pred, target.squeeze(1))
        vloss += loss.item()
        pbar.set_postfix({
            'iter loss': loss.item(),
            'val loss': vloss / (batch_idx + 1)
        })
        accelerator.log({
            'val loss': vloss / (batch_idx + 1),
            'val iter loss': loss.item(),
            'step': kwargs['step']
        }, step=n_epoch * loader_tot + batch_idx + 1)
    return vloss / (batch_idx + 1)


if __name__ == '__main__':
    import datetime as dt
    now = dt.datetime.now()
    pdir = f'./static/log/hib/{now:%Y-%m-%d_%H-%M-%S}'
    config = {
        'utils_config': {
            'need_channels': True,
            'need_weights': True,
            'categories_names': HU.FULL_CATEGORIES,
            'auto_categories': True,
            'phase': 'train',
            'window': [200, 300],
        },
        'n_epoch': 300,
        'batch_size': {'train': 1024, 'val': 1024},
        'pconfig': dict(project_dir=pdir, logging_dir=f'{pdir}/board'),
        'nc': 23,
        'model': 'resnet50-1d',
        'optimizer': 'adamw',

    }

    pconfig = accelerate.utils.ProjectConfiguration(
        **config['pconfig']
    )
    accelerator = accelerate.Accelerator(project_config=pconfig, log_with=['tensorboard'])
    accelerator.init_trackers('HIB Classifier')
    import json
    with open(f'{accelerator.project_dir}/config.json', 'w+') as jout:
        json.dump(config, jout)
    config['status'] = 'OK'
    try:
        main(config)
    except Exception as e:
        config['status'] = f'ERROR:\n{e.args}'
        with open(f'{accelerator.project_dir}/config.json', 'w+') as jout:
            json.dump(config, jout)
        import traceback
        traceback.print_exc()

    with open(f'{accelerator.project_dir}/config.json', 'w+') as jout:
        json.dump(config, jout)
