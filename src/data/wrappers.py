from typing import Iterable, Any


class Batcher:
    def __init__(self, inner: Iterable[Any], batch_size: int):
        self.inner = inner
        self.batch_size = batch_size
        self.buffer = [0] * batch_size
        self.buffer_idx = 0

    def __iter__(self):
        for item in self.inner:
            self.buffer[self.buffer_idx] = item
            self.buffer_idx += 1
            if self.buffer_idx == self.batch_size:
                yield self.buffer
                self.buffer_idx = 0
        self.reset()

    def reset(self):
        self.buffer = [0] * self.batch_size
        self.buffer_idx = 0
