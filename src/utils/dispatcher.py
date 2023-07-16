import multiprocessing as mp


def dispatch_worker(shard_lists, num_ddp):
    dispatcher = Dispatcher(shard_lists, num_ddp)
    dispatcher.start()
    dispatcher.join()


class Dispatcher:
    def __init__(self, shard_lists, num_ddp):
        self.shard_lists = shard_lists
        self.num_ddp = num_ddp
        self.queues = [mp.Queue() for _ in range(num_ddp)]
        self.processes = []

    def start(self):
        ctx = mp.get_context('spawn')
        for i, sl in enumerate(self.shard_lists):
            p = ctx.Process(target=reader_worker, args=(sl, self.queues, i, self.num_ddp))
            p.start()
            self.processes.append(p)

    def join(self):
        for p in self.processes:
            p.join()
        for queue in self.queues:
            queue.put(StopIteration)


def reader_worker(shard_list, queues, start_queue_idx, num_queues):
    from .datapipes import TarToWDS
    from itertools import cycle

    reader = TarToWDS(shard_list)
    queue_order = list(range(start_queue_idx, num_queues)) + list(range(start_queue_idx))

    for item, queue_idx in zip(reader, cycle(queue_order)):
        queues[queue_idx].put(item)
