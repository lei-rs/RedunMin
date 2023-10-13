import optax
import haliax as hax
import jax.random as jax_rand

from src.data import DLConfig
from src.data.loader import DataLoader
from src.model.lq import LQViTConfig, LQViT
from src.trainer import TrainerConfig, Trainer

Batch = hax.Axis(name='batch', size=64)


def bce_loss(y_hat, y):
    return optax.sigmoid_binary_cross_entropy(y_hat, y).sum(axis=-1).mean()


if __name__ == '__main__':
    dl_cfg = DLConfig(
        data_loc='gs://redunmin-us',
        batch_size=Batch.size,
        shuffle=True,
        n_frames=32,
        base_seed=42,
    )
    dl = DataLoader('ssv2', dl_cfg)

    key = jax_rand.PRNGKey(0)
    key, key_model = jax_rand.split(key)
    cfg = LQViTConfig()
    model = LQViT.from_pretrained(
        'google/vit-base-patch16-224',
        f'{dl_cfg.data_loc}/vit-base-16-224.safetensors',
        cfg,
        key=key_model,
    )

    train_cfg = TrainerConfig(
        max_epochs=200,
        loss_fn=bce_loss,
        optim_cfg={
            'learning_rate': 1e-4,
            'weight_decay': 1e-2,
        },
        optim=optax.adamw,
    )
    trainer = Trainer(train_cfg, model, dl)
    trainer.train()
