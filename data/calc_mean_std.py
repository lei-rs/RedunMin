import io
import torch
import numpy as np
import webdataset as wds
from tqdm import tqdm
from PIL import Image


torch.multiprocessing.set_sharing_strategy('file_system')


def assemble_frames(sample):
    frames = []
    num_frames = int(sample['num_frames'])
    frame_indices = np.arange(0, num_frames, num_frames / 128).astype(int)

    for i in frame_indices:
        stream = io.BytesIO(sample[f'frame_{i:06d}.jpeg'])
        img = np.asarray(Image.open(stream), dtype=np.float16)
        frames.append(img / 255.)

    return np.mean(frames, axis=(0, 1, 2)), np.std(frames, axis=(0, 1, 2))


urls = [f'https://actionreg-data.s3.amazonaws.com/ssv2/shard_{i:06d}.tar' for i in range(34)]
ds = wds.WebDataset(urls).map(assemble_frames)
loader = wds.WebLoader(ds, batch_size=None, num_workers=32)

m = []
s = []
for i, x in tqdm(enumerate(loader)):
    m.append(x[0].numpy())
    s.append(x[1].numpy())

    if i % 1000 == 0:
        print(np.mean(np.asarray(m), axis=0))
        print(np.mean(np.asarray(s), axis=0))
