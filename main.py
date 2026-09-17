"""Puzzle Decoder Race solver.

Keeps a fixed number of fragment requests in flight, reassembling the message
as soon as enough replies have arrived to be statistically confident that no
fragment is still missing. Stdout carries only the final message; run
diagnostics go to stderr.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Iterator
from random import Random

# Started before importing httpx so the reported total includes its load cost.
_STARTED_AT = time.perf_counter()

import httpx  # noqa: E402

BASE_URL = os.environ.get("PUZZLE_SERVER_URL", "http://127.0.0.1:8080")

# Requests kept in flight at once. Chosen to stay inside the server's
# low-latency zone without overloading its connection queue.
WINDOW = 120

# Leading block of consecutive ids, each requested REPLICAS times so the
# fastest reply for each one is used instead of a single random delay.
CONSECUTIVE_SPAN = 48
REPLICAS = 3

# Probability tolerated of declaring the puzzle complete while a fragment is
# still missing.
CONFIDENCE_DELTA = 1e-3

# Hard budgets so the program always terminates.
MAX_PROBES = 800
MAX_WALL_SECONDS = 5.0

PROBE_STRIDE = 9973
PROBE_RANDOM_CEILING = 10**9
PROBE_SEED = 20260917

# Fragments are concatenated directly.
FRAGMENT_SEPARATOR = " "

LIMITS = httpx.Limits(max_connections=WINDOW, max_keepalive_connections=WINDOW)
TIMEOUT = httpx.Timeout(2.0, connect=1.0)


class FragmentStore:
    """Tracks fragments received and the stats needed to detect completion."""

    def __init__(self) -> None:
        self._texts: dict[int, str] = {}
        self._counts: Counter[int] = Counter()
        self.responses = 0

    def add(self, index: int, text: str) -> None:
        """Record a fragment reply, deduplicated by index."""
        self._texts.setdefault(index, text)
        self._counts[index] += 1
        self.responses += 1

    @property
    def distinct(self) -> int:
        """Number of distinct fragment indices seen so far."""
        return len(self._texts)

    @property
    def singletons(self) -> int:
        """Number of indices seen exactly once (Good-Turing f1 statistic)."""
        return sum(1 for count in self._counts.values() if count == 1)

    @property
    def missing_mass(self) -> float:
        """Good-Turing estimate of the probability mass still unseen."""
        if self.responses == 0:
            return 1.0
        return self.singletons / self.responses

    def is_contiguous_from_zero(self) -> bool:
        """Whether seen indices form an unbroken range starting at 0."""
        return bool(self._texts) and max(self._texts) == self.distinct - 1

    def message(self) -> str:
        """Assemble the collected fragments in index order."""
        return FRAGMENT_SEPARATOR.join(text for _, text in sorted(self._texts.items()))


class RunMetrics:
    """Timings and counters collected for the diagnostic report."""

    def __init__(self) -> None:
        self.probes_launched = 0
        self.failures = 0
        self.first_response_at: float | None = None
        self.solved_at: float | None = None


def min_samples_for_confidence(distinct: int, delta: float) -> int:
    """Minimum replies needed to rule out a missing fragment at the given confidence level.

    Assumes a puzzle of size distinct + 1 in the worst case.
    """
    if distinct < 1:
        raise ValueError("at least one fragment must have been observed")
    return math.ceil(math.log(delta) / math.log(distinct / (distinct + 1)))


def is_complete(store: FragmentStore) -> bool:
    """Whether the evidence collected justifies declaring the puzzle done.

    Requires a contiguous run of indices from zero, enough replies to make a
    missing fragment statistically implausible, and no fragment seen only
    once (a fresh singleton would suggest more pieces are still surfacing).
    """
    if not store.is_contiguous_from_zero():
        return False
    if store.responses < min_samples_for_confidence(store.distinct, CONFIDENCE_DELTA):
        return False
    return store.singletons == 0


def probe_id_stream() -> Iterator[int]:
    """Yield an endless schedule of fragment ids to probe.

    Leads with a replicated block of consecutive ids for fast, low-latency
    coverage, then falls back to a mix of consecutive, strided, and random
    ids so the schedule keeps working regardless of how the server maps ids
    to fragments.
    """
    for _ in range(REPLICAS):
        yield from range(CONSECUTIVE_SPAN)
    rng = Random(PROBE_SEED)
    consecutive = itertools.count(CONSECUTIVE_SPAN)
    strided = (step * PROBE_STRIDE for step in itertools.count(1))
    while True:
        yield next(consecutive)
        yield next(strided)
        yield rng.randrange(PROBE_RANDOM_CEILING)
        yield rng.randrange(PROBE_RANDOM_CEILING)


async def fetch_fragment(
    client: httpx.AsyncClient, fragment_id: int
) -> tuple[int, str] | None:
    """Fetch one fragment, returning None on any failure.

    Failures are dropped rather than retried: with each id probed several
    times over, redundancy already covers the occasional bad request.
    """
    try:
        response = await client.get("/fragment", params={"id": fragment_id})
        response.raise_for_status()
        payload = response.json()
        return int(payload["index"]), str(payload["text"])
    except (httpx.HTTPError, ValueError, KeyError):
        return None


async def solve(client: httpx.AsyncClient) -> tuple[FragmentStore, RunMetrics, bool]:
    """Assemble the puzzle, holding WINDOW requests in flight throughout."""
    store = FragmentStore()
    metrics = RunMetrics()
    ids = probe_id_stream()
    started = time.perf_counter()
    in_flight: set[asyncio.Task[tuple[int, str] | None]] = set()

    try:
        while True:
            while len(in_flight) < WINDOW and metrics.probes_launched < MAX_PROBES:
                in_flight.add(asyncio.create_task(fetch_fragment(client, next(ids))))
                metrics.probes_launched += 1

            if not in_flight or time.perf_counter() - started > MAX_WALL_SECONDS:
                return store, metrics, False

            done, in_flight = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                result = task.result()
                if result is None:
                    metrics.failures += 1
                    continue
                if metrics.first_response_at is None:
                    metrics.first_response_at = time.perf_counter()

                store.add(*result)

            if is_complete(store):
                metrics.solved_at = time.perf_counter()
                return store, metrics, True
    finally:
        for task in in_flight:
            task.cancel()


def build_client(
    base_url: str = BASE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build the HTTP client used to talk to the puzzle server."""
    return httpx.AsyncClient(
        base_url=base_url,
        transport=transport or httpx.AsyncHTTPTransport(retries=0, limits=LIMITS),
        timeout=TIMEOUT,
    )


def report(store: FragmentStore, metrics: RunMetrics, solved: bool) -> None:
    """Print run diagnostics to stderr."""
    now = time.perf_counter()
    first = metrics.first_response_at or _STARTED_AT
    print(
        "\n".join(
            [
                f"probes launched     : {metrics.probes_launched}",
                f"replies used        : {store.responses}",
                f"failed probes       : {metrics.failures}",
                f"distinct fragments  : {store.distinct}",
                f"unobserved mass (GT): {store.missing_mass:.4f}",
                f"time to first reply : {first - _STARTED_AT:.3f} s",
                f"time in flight      : {(metrics.solved_at or now) - first:.3f} s",
                f"total wall clock    : {now - _STARTED_AT:.3f} s",
                f"complete            : {'yes' if solved else 'no'}",
            ]
        ),
        file=sys.stderr,
    )


async def main() -> int:
    """Run the solver and print the assembled message and diagnostics."""
    client = build_client()
    try:
        store, metrics, solved = await solve(client)
        # Printed before teardown so cleanup never delays showing the result.
        print(store.message(), flush=True)
        report(store, metrics, solved)
        return 0 if solved else 1
    finally:
        await client.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
