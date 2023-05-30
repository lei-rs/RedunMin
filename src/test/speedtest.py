import os
from src import utils as U
import time


ffm = os.environ['SM_CHANNEL_TRAINING']
path = [os.path.join(ffm, f'ssv2/train/shard_{i:06d}.tar') for i in range(47)]

transforms = [
    U.SampleFrames(128, 'uniform'),
    U.ReadFrames(),
]

dl = U.Dataloader(path, batch_size=1, num_workers=4, transforms=transforms, prefetch_count=8).dl

start = time.time()
for i, batch in enumerate(dl):
    continue

print(f'Time taken: {time.time() - start:.2f}s \n'
      f'Iterations: {i + 1} \n'
      f'FPS: {(i + 1) / (time.time() - start):.2f}')
