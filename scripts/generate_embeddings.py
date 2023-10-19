from typing import Iterable, Tuple

import torch
import pickle
import open_clip
import torch_xla.core.xla_model as xm
from tqdm import tqdm
from torchvision.transforms.v2 import Normalize, ToDtype
from rand_archive import Reader, Writer

from src.data.types import VideoSample


model, pre_proc = open_clip.create_model_from_pretrained(
    'ViT-B-16',
    'laion2b-s34b-b88K',
    device=xm.xla_device(),
    jit=True,
)
image_mean = getattr(model.visual, 'image_mean', None)
image_std = getattr(model.visual, 'image_std', None)
norm = Normalize(image_mean, image_std, inplace=True)
cast = ToDtype(torch.float32, scale=True)


def generate_embeddings(dataset: Iterable) -> Iterable[Tuple[str, torch.Tensor]]:
    for k, v in dataset:
        _, v = VideoSample(**pickle.loads(v)).to_arrays()
        v = torch.stack([torch.from_numpy(f) for f in v]).to(xm.xla_device())
        with torch.no_grad():
            yield k, model.encode_image(norm(cast(v)))


if __name__ == '__main__':
    r = Reader().open_gcs('gs://redunmin-us/ssv2/train.raa')
    w = Writer('ssv2/train_embeddings.raa')
    for k, e in tqdm(generate_embeddings(r)):
        e = e.numpy()
        w.write(k, bytes(e.dumps()))
