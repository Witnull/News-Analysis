import asyncio
import gc
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Iterator


class LLMPool:
    """
    Runs N instances of an LLM_Base subclass (e.g. llmFeedParser) side by side
    for "fake parallel" inference without a server - each instance owns its
    own VRAM allocation, round-robin assigns work, and each instance is
    individually semaphore-guarded to avoid concurrent calls on the same model.
    """

    def __init__(
        self, llm_cls, n_workers: int = 2, verbose: bool = False, **llm_kwargs
    ):
        self.llm_cls = llm_cls
        self.n_workers = n_workers
        self.verbose = verbose
        self.llm_kwargs = llm_kwargs

        self._instances: List = []
        self._semaphores: List[asyncio.Semaphore] = []
        self._executor: ThreadPoolExecutor | None = None
        self._cycle: Iterator[int] | None = None

    def load(self) -> "LLMPool":
        if self._instances:
            return self

        if self.verbose:
            print(
                f"[LLMPool] Loading {self.n_workers} instance(s) of {self.llm_cls.__name__}"
            )
            t0 = time.time()

        for i in range(self.n_workers):
            inst = self.llm_cls(**self.llm_kwargs).load()
            self._instances.append(inst)
            self._semaphores.append(asyncio.Semaphore(1))

        self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        self._cycle = itertools.cycle(range(self.n_workers))

        if self.verbose:
            print(
                f"[LLMPool] Ready in {time.time() - t0:.1f}s ({self.n_workers} workers)"
            )

        return self

    def unload(self):
        for inst in self._instances:
            inst.unload()
        self._instances.clear()
        self._semaphores.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._cycle = None
        gc.collect()
        if self.verbose:
            print("[LLMPool] All instances unloaded.")

    def _next_idx(self) -> int:
        if self._cycle is None:
            raise RuntimeError("LLMPool.load() must be called before use")
        return next(self._cycle)

    async def call(self, method_name: str, *args, **kwargs):
        """Round-robins across loaded instances; each instance is single-flight.

        method_name: name of the method to call on the underlying instance
        (e.g. "parse_article", "classify", "summarize", etc.)
        """
        idx = self._next_idx()
        loop = asyncio.get_running_loop()
        async with self._semaphores[idx]:
            instance = self._instances[idx]
            method = getattr(instance, method_name)
            return await loop.run_in_executor(
                self._executor,
                lambda: method(*args, **kwargs),
            )