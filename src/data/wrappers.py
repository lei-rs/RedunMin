from typing import Iterable, Any


class Batcher:
    def __init__(self, inner: Iterable[Any], n_items: int, batch_size: int):
        self.inner = inner
        self.n_items = n_items
        self.batch_size = batch_size
        self.buffer = [[0] * batch_size for _ in range(n_items)]
        self.buffer_idx = 0

    def __iter__(self):
        for item in self.inner:
            assert len(item) == self.n_items, f'Expected {self.n_items} items, got {len(item)}'
            for i in range(self.n_items):
                self.buffer[i][self.buffer_idx] = item[i]
            self.buffer_idx += 1
            if self.buffer_idx == self.batch_size:
                yield tuple(self.buffer)
                self.buffer = [[0] * self.batch_size for _ in range(self.n_items)]
                self.buffer_idx = 0
        self.reset()

    def reset(self):
        self.buffer = [0] * self.batch_size
        self.buffer_idx = 0
