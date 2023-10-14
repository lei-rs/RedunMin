import optax
import haliax as hax
import jax.random as jax_rand
from jax.numpy import bfloat16

from src.data.loader import DLConfig, SSV2
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
    dl = SSV2(dl_cfg)

    key = jax_rand.PRNGKey(0)
    key, key_model, key_trainer = jax_rand.split(key, 3)
    cfg = LQViTConfig()
    model = LQViT.from_pretrained(
        'google/vit-base-patch16-224',
        f'{dl_cfg.data_loc}/vit/vit-base-16-224.safetensors',
        cfg,
        key=key_model,
        dtype=bfloat16,
    )

    train_cfg = TrainerConfig(
        max_epochs=200,
        loss_fn=bce_loss,
        optim=optax.adamw(
            1e-4,
            weight_decay=1e-2,
        ),
    )
    trainer = Trainer(train_cfg, model, dl, key=key_trainer)
    trainer.train()
