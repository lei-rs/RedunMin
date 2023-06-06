import os
import time

from src.utils import SSv2


path = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'ssv2')
dm = SSv2(64, False, 10, 'ssv2')
dm.prepare_data()
dm.setup('train')
dl = dm.train_dataloader()


start = time.time()
for i, batch in enumerate(dl):
    continue

print(f'Time taken: {time.time() - start:.2f}s \n'
      f'Iterations: {i + 1} \n'
      f'FPS: {(i + 1) / (time.time() - start):.2f}')
