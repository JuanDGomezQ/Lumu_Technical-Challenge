"""Tests for the puzzle solver.

Uses a fake httpx transport in place of the Docker server so the suite runs
offline and deterministically, while still exercising the real client, the
real concurrency, and the real stopping rule.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from random import Random
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from main import (
    CONFIDENCE_DELTA,
    FRAGMENT_SEPARATOR,
    FragmentStore,
    build_client,
    is_complete,
    min_samples_for_confidence,
    probe_id_stream,
    solve,
)


class FakePuzzleServer(httpx.AsyncBaseTransport):
    """Emulates the challenge server: fixed puzzle, random delay, any id valid."""

    def __init__(
        self,
        fragments: list[str],
        *,
        delay_range: tuple[float, float] = (0.0, 0.0),
        failure_rate: float = 0.0,
        skew: bool = False,
        concurrency: int | None = None,
        seed: int = 1234,
    ) -> None:
        """Initialize the fake puzzle server with configurable network behavior."""
        self._fragments = fragments
        self._delay_range = delay_range
        self._failure_rate = failure_rate
        self._skew = skew
        self._rng = Random(seed)
        self._semaphore = asyncio.Semaphore(concurrency) if concurrency else None

    def _index_for(self, fragment_id: int) -> int:
        """Map a requested fragment ID to a puzzle index based on server mode."""
        if self._skew and len(self._fragments) > 1:
            if fragment_id % 50 == 0:
                return len(self._fragments) - 1
            return fragment_id % (len(self._fragments) - 1)
        return fragment_id % len(self._fragments)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Simulate processing an HTTP GET request for a puzzle fragment."""
        low, high = self._delay_range
        if high > 0:
            if self._semaphore is not None:
                async with self._semaphore:
                    await asyncio.sleep(self._rng.uniform(low, high))
            else:
                await asyncio.sleep(self._rng.uniform(low, high))

        if self._rng.random() < self._failure_rate:
            return httpx.Response(500, request=request)

        fragment_id = int(parse_qs(urlparse(str(request.url)).query)["id"][0])
        index = self._index_for(fragment_id)
        body = json.dumps(
            {"id": fragment_id, "index": index, "text": self._fragments[index]}
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
            request=request,
        )


def run_solver(server: FakePuzzleServer):
    """Execute the puzzle solver against the given fake server transport."""

    async def runner():
        client = build_client(base_url="http://puzzle.test", transport=server)
        try:
            return await solve(client)
        finally:
            await client.aclose()

    return asyncio.run(runner())


def store_with(counts: dict[int, int]) -> FragmentStore:
    """Create a FragmentStore pre-populated with specific piece counts."""
    store = FragmentStore()
    for index, count in counts.items():
        for _ in range(count):
            store.add(index, f"f{index}")
    return store


# Confidence bound


def test_min_samples_matches_closed_form():
    """Verify minimum sample calculations against analytical bounds."""
    assert min_samples_for_confidence(1, 0.5) == 1
    assert min_samples_for_confidence(1, 0.25) == 2
    assert min_samples_for_confidence(10, CONFIDENCE_DELTA) == 73
    assert min_samples_for_confidence(30, CONFIDENCE_DELTA) == 211


def test_min_samples_grows_with_puzzle_size():
    """Ensure the sample threshold monotonically increases with puzzle size."""
    thresholds = [min_samples_for_confidence(k, CONFIDENCE_DELTA) for k in range(1, 35)]
    assert thresholds == sorted(thresholds)


def test_min_samples_rejects_empty_store():
    """Verify that zero distinct fragments raises a ValueError."""
    with pytest.raises(ValueError):
        min_samples_for_confidence(0, CONFIDENCE_DELTA)


# Stopping rule


def test_gap_is_never_complete():
    """Ensure completion fails if fragment indices contain gaps."""
    assert not is_complete(store_with({0: 500, 1: 500, 3: 500}))


def test_empty_store_is_never_complete():
    """Ensure an empty store is never declared complete."""
    assert not is_complete(FragmentStore())


def test_contiguous_but_undersampled_is_not_complete():
    """Ensure completion fails if total sample count is below confidence threshold."""
    assert not is_complete(store_with({0: 1, 1: 1, 2: 1}))


def test_singletons_block_completion():
    """Ensure completion fails if any fragment has only been seen once (f1 > 0)."""
    counts = {index: 40 for index in range(3)}
    counts[3] = 1
    assert not is_complete(store_with(counts))


def test_well_sampled_contiguous_store_is_complete():
    """Ensure completion succeeds when contiguous fragments have sufficient depth."""
    assert is_complete(store_with({index: 20 for index in range(10)}))


# Probe schedule


def test_schedule_is_deterministic():
    """Verify that the probing schedule yields identical ID sequences."""
    assert list(itertools.islice(probe_id_stream(), 400)) == list(
        itertools.islice(probe_id_stream(), 400)
    )


def test_tail_covers_multiple_id_strategies():
    """Verify that the schedule reaches pseudo-random high IDs."""
    tail = list(itertools.islice(probe_id_stream(), 600))
    assert any(value > 10**6 for value in tail)


# End to end


def test_assembles_ten_fragment_puzzle():
    """Test end-to-end resolution of a 10-piece puzzle."""
    fragments = [f"part{index}" for index in range(10)]
    store, _, solved = run_solver(FakePuzzleServer(fragments))
    assert solved
    assert store.message() == FRAGMENT_SEPARATOR.join(fragments)


def test_assembles_single_fragment_puzzle():
    """Test end-to-end resolution of a single-piece puzzle."""
    store, _, solved = run_solver(FakePuzzleServer(["only"]))
    assert solved
    assert store.message() == "only"


def test_assembles_thirty_fragment_puzzle():
    """Test end-to-end resolution of a standard 30-piece puzzle."""
    fragments = [f"w{index:02d}" for index in range(30)]
    store, _, solved = run_solver(FakePuzzleServer(fragments))
    assert solved
    assert store.message() == FRAGMENT_SEPARATOR.join(fragments)


def test_survives_skewed_distribution():
    """Test solver recovery when one piece is disproportionately rare."""
    fragments = [f"p{index}" for index in range(6)]
    store, _, solved = run_solver(FakePuzzleServer(fragments, skew=True))
    assert solved
    assert store.distinct == len(fragments)


def test_survives_flaky_server():
    """Test solver resilience against random HTTP 500 errors."""
    fragments = [f"w{index:02d}" for index in range(30)]
    store, metrics, solved = run_solver(FakePuzzleServer(fragments, failure_rate=0.2))
    assert solved
    assert metrics.failures > 0
    assert store.message() == FRAGMENT_SEPARATOR.join(fragments)


def test_finishes_within_one_round_trip_under_realistic_latency():
    """Verify performance and completion under simulated network delays."""
    fragments = [f"w{index:02d}" for index in range(30)]
    server = FakePuzzleServer(fragments, delay_range=(0.1, 0.4), concurrency=100)
    begin = time.perf_counter()
    store, _, solved = run_solver(server)
    elapsed = time.perf_counter() - begin
    assert solved
    assert store.message() == FRAGMENT_SEPARATOR.join(fragments)
    assert elapsed < 2.0
