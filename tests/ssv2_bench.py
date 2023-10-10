from tqdm import tqdm

from src.data import DLConfig, SSV2

config = DLConfig(
    data_loc='gs://redunmin-us',
    batch_size=64,
)
loader = SSV2(config)

for i, x in enumerate(tqdm(loader.train_loader())):
    pass