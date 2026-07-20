'''

'''

import time
from contextlib import contextmanager

import torch


class PerfTimer:
    """Accumulates elapsed wall-clock seconds per named phase across repeated calls."""

    def __init__(self):
        self._sums = {}

    @contextmanager
    def time(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._sums[name] = self._sums.get(name, 0.0) + elapsed

    def as_metrics(self, prefix):
        return {f"{prefix}/{name}_seconds": value for name, value in self._sums.items()}

    def reset(self):
        self._sums = {}


def gpu_mem_snapshot(prefix):
    if not torch.cuda.is_available():
        return {}
    return {
        f"{prefix}/mem_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        f"{prefix}/mem_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        f"{prefix}/max_mem_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }
