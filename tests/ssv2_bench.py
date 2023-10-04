from src.data import DLConfig, SSV2
from tqdm import tqdm


config = DLConfig(
    data_loc='gs://redunmin',
    batch_size=32,
    _sim_shard=(0, 8)
)
loader = SSV2(config)

for i in tqdm(loader.train_loader()):
    pass