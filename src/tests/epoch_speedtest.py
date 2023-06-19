import os

import lightning as L
import torch
from torch import nn

from src.utils import SSv2


class BaseModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.randn(1))

    def training_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        return None


if __name__ == '__main__':
    #path = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'ssv2')
    path = 'C:\\Users\leifu\Documents\GitHub\RedunMin\data\ssv2'
    data = SSv2(64, True, 10, False, path)
    trainer = L.Trainer(accelerator='cpu', devices=2, strategy='ddp', max_epochs=1)
    trainer.fit(BaseModule(), datamodule=data)
