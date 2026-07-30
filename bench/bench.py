"""Benchmark Mojo transition resolution against pytz on the same data."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta
import math
import os
import platform
import sys
import time

import numpy as np
import pytz

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojopytz  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def benchmark_offsets(values, mojo_zone, pytz_zone):
    transition_seconds = [
        (value - datetime(1970, 1, 1)).days * 86400
        + (value - datetime(1970, 1, 1)).seconds
        for value in pytz_zone._utc_transition_times
    ]
    offsets = [int(info[0].total_seconds()) for info in pytz_zone._transition_info]

    def reference():
        result = np.empty(values.size, dtype=np.int64)
        for index, value in enumerate(values):
            period = max(0, bisect_right(transition_seconds, int(value)) - 1)
            result[index] = offsets[period]
        return result

    assert np.array_equal(mojo_zone.offsets_at(values), reference())
    mojo_zone.offsets_at(values)
    return timeit(lambda: mojo_zone.offsets_at(values)), timeit(reference)


def benchmark_localize(values, mojo_zone, pytz_zone):
    def reference():
        return [pytz_zone.localize(value, is_dst=False) for value in values]

    got = mojo_zone.localize_many(values, is_dst=False)
    ref = reference()
    assert [value.utcoffset() for value in got] == [
        value.utcoffset() for value in ref
    ]
    mojo_zone.localize_many(values, is_dst=False)
    return timeit(
        lambda: mojo_zone.localize_many(values, is_dst=False)
    ), timeit(reference)


def main():
    mojo_zone = mojopytz.timezone("US/Eastern")
    pytz_zone = pytz.timezone("US/Eastern")
    count = 250_000
    sorted_values = np.linspace(
        -2_000_000_000, 2_000_000_000, count, dtype=np.int64
    )
    shuffled_values = np.random.default_rng(0).permutation(sorted_values)
    local_values = [
        datetime(2000, 1, 1) + timedelta(minutes=15 * index)
        for index in range(100_000)
    ]

    rows = [
        (
            "UTC offsets, sorted (250k)",
            *benchmark_offsets(sorted_values, mojo_zone, pytz_zone),
        ),
        (
            "UTC offsets, shuffled (250k)",
            *benchmark_offsets(shuffled_values, mojo_zone, pytz_zone),
        ),
        (
            "localize, ordinary times (100k)",
            *benchmark_localize(local_values, mojo_zone, pytz_zone),
        ),
    ]

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | mojo-pytz | pytz reference | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, pytz_time in rows:
        ratio = pytz_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{pytz_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
