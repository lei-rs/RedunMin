from typing import Iterable

import jax


class Batcher:
    def __init__(self, inner: Iterable, batch_size: int, n_items: int = 2):
        self.inner = inner
        self.batch_size = batch_size
        self.n_items = n_items
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


class Stopper:
    def __init__(self, inner: Iterable):
        self.inner = inner

    def __iter__(self):
        for item in self.inner:
            stop_signal = jax.device_put(0)
            summed = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(stop_signal)
            if jax.block_until_ready(summed) > 0:
                return
            yield item
        stop_signal = jax.device_put(1)
        _ = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(stop_signal)
