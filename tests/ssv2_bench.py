import jax
from tqdm import tqdm

from src.data import DLConfig, SSV2
from src.data.loader import DataLoader

config = DLConfig(
    data_loc='gs://redunmin-us',
    batch_size=1,
)
loader = SSV2(config)

for i, x in enumerate(tqdm(loader.train_loader())):
    if i > 10_000:
        break