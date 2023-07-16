import os
import time
from torchdata.datapipes.iter import IterableWrapper
from multiprocessing import Pool
import src.utils.datapipes as udp


def load_shard(fp):
    os.chdir(os.path.expanduser('C:\\Users\leifu\Documents\GitHub\RedunMin'))
    with open(fp, 'rb') as f:
        f.read(-1)
    return


if __name__ == '__main__':
    p = Pool(2)
    #path = os.path.join(os.environ['SM_CHANNEL_TRAINING'])
    os.chdir(os.path.expanduser('C:\\Users\leifu\Documents\GitHub\RedunMin'))
    fp = [f'data/ssv2/train/shard_{i:06d}.tar' for i in range(2)]
    #fp = [os.path.join(path, f) for f in os.listdir(path)]
    start = time.time()
    p.map(load_shard, fp)
    print(f'Time taken: {time.time() - start:.2f}s')
    p.close()
    p.join()
