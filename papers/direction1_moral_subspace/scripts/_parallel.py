"""Ordered, fault-tolerant parallel map over API-bound work.

ThreadPoolExecutor is the right tool here: the work is blocking network I/O (Anthropic
calls), so threads overlap latency without GIL contention. Results are returned in input
order; an item whose callable raises is recorded in ``errors`` and left as ``None`` in the
results list (so a few transient failures never abort a long run).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def parallel_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    workers: int,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[Any], dict[int, str]]:
    """Apply *fn* to each item across *workers* threads, preserving order.

    Returns ``(results, errors)`` where ``results[i]`` is ``fn(items[i])`` or ``None`` if
    it raised, and ``errors`` maps the failed index to a truncated message.
    """
    results: list[Any] = [None] * len(items)
    errors: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_i = {ex.submit(fn, x): i for i, x in enumerate(items)}
        done = 0
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 -- record and continue the run
                errors[i] = str(e)[:200]
            done += 1
            if on_progress and (done % 25 == 0 or done == len(items)):
                on_progress(done, len(items), len(errors))
    return results, errors
