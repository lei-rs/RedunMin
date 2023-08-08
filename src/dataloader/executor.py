from queue import Queue
from threading import Thread, Lock
from typing import Iterable, Callable, Optional


class ThreadSafeIterator:
    def __init__(self, it):
        self._it = it
        self.lock = Lock()

    def __iter__(self):
        self._it = iter(self._it)
        return self

    def __next__(self):
        with self.lock:
            return next(self._it)


class LazyThreadPoolExecutor:
    def __init__(self, max_workers: int, queue_size=16):
        self._max_workers = max_workers
        self._workers = []
        self._output_q = Queue(queue_size)

        self.inputs: Optional[Iterable] = None
        self.func: Optional[Callable] = None
        self.running = False

    def _worker(self):
        for item in self.inputs:
            self._output_q.put(self.func(item), block=True)
        self._output_q.put(StopIteration, block=True)

    def _start(self):
        for i in range(self._max_workers):
            worker = Thread(target=self._worker, daemon=True)
            worker.start()
            self._workers.append(worker)

    def stop(self):
        for worker in self._workers:
            worker.join(timeout=0.01)

    def map(self, func: Callable, inputs: Iterable):
        assert not self.running, "Executor is already running"
        self.func = func
        self.inputs = ThreadSafeIterator(inputs)

    def __iter__(self):
        assert self.func is not None and self.inputs is not None, "Executor not initialized"

        done = 0
        self._start()
        self.running = True

        while True:
            item = self._output_q.get(block=True)
            if item is StopIteration:
                done += 1
                if done == self._max_workers:
                    break
                else:
                    continue
            else:
                yield item

        self.running = False
