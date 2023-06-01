import os
from typing import Optional, Literal

import lightning as L
import torch
from torch import nn

from src.utils import SSv2


class BaseModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.randn(1))

    def training_step(self, batch, batch_idx):
        return Literal

    def configure_optimizers(self):
        return None


if __name__ == '__main__':
    os.chdir(os.path.expanduser('/home/lei/Documents/RedunMin/data'))
    data = SSv2(64, True, 10, 'ssv2')
    trainer = L.Trainer(accelerator='gpu', strategy='ddp', fast_dev_run=True)
    trainer.fit(BaseModule(), datamodule=data)
