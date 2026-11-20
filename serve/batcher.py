import asyncio


class MicroBatcher:
    """Dynamic batching: collects concurrent requests for up to `max_wait_ms` (or `max_batch` items)
    and runs them through the model as ONE batch -> far higher GPU/CPU throughput under load."""
    def __init__(self, fn, max_batch=16, max_wait_ms=8):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self.q, self.task = None, None
        self.batches = self.items = 0

    async def start(self):
        self.q = asyncio.Queue(); self.task = asyncio.create_task(self._loop())

    async def stop(self):
        if self.task:
            self.task.cancel()

    async def submit(self, x):
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((x, fut))
        return await fut                       # -> (output_row, batch_size)

    async def _loop(self):
        import torch
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.q.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch and (left := deadline - loop.time()) > 0:
                try:
                    batch.append(await asyncio.wait_for(self.q.get(), left))
                except asyncio.TimeoutError:
                    break
            self.batches += 1; self.items += len(batch)
            try:
                out = await loop.run_in_executor(None, self.fn, torch.stack([x for x, _ in batch]))
                for (_, f), o in zip(batch, out):
                    if not f.done(): f.set_result((o, len(batch)))
            except Exception as e:
                for _, f in batch:
                    if not f.done(): f.set_exception(e)

    @property
    def mean_batch(self):
        return round(self.items / self.batches, 2) if self.batches else 0.0
