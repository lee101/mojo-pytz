# mojo-pytz

`mojo-pytz` is a focused port of pytz's timezone offset-resolution hot path to
[Mojo](https://www.modular.com/mojo). It keeps pytz's familiar scalar API while
adding batch methods that resolve large arrays of UTC timestamps or local wall
times in one compiled call.

```python
from datetime import datetime
import numpy as np
import mojopytz as pytz

eastern = pytz.timezone("US/Eastern")

# The covered scalar API has pytz's names and signatures.
aware = eastern.localize(datetime(2020, 11, 1, 1, 30), is_dst=False)
print(aware)  # 2020-11-01 01:30:00-05:00

# Integer inputs are POSIX seconds on the UTC axis.
seconds = np.array([0, 1_000_000_000, 1_600_000_000], dtype=np.int64)
print(eastern.offsets_at(seconds))  # [-18000 -14400 -14400]

# datetime64 arrays retain their shape.
times = np.array(["2020-01-01", "2020-07-01"], dtype="datetime64[s]")
print(eastern.tzname_at(times))  # ['EST' 'EDT']
```

## Coverage

The Mojo implementation covers the work that benefits from compilation:

- transition-table lookup for sorted or unsorted UTC timestamps;
- UTC offset, DST adjustment, abbreviation, and transition-index batches;
- UTC-to-local conversion with `fromutc_many`;
- local wall-time resolution with `localize_many`, including ambiguous folds,
  nonexistent gaps, negative DST, half-hour changes, and historical
  same-DST transitions.

The Python surface also provides pytz-compatible `timezone(zone)`,
`localize(dt, is_dst=False)`, `normalize(dt)`, `fromutc(dt)`, `utcoffset`,
`dst`, and `tzname`. It re-exports `UTC`, `utc`, `FixedOffset`, pytz's
exception classes, timezone lists, and country mappings. Existing scalar code
can generally switch from `import pytz` to `import mojopytz as pytz`.

This project deliberately does not duplicate pytz's bundled IANA database,
TZif parser, lazy dictionary implementation, or pickle compatibility
internals. The installed `pytz` package supplies that data and the mature
scalar implementations. Mojo performs the covered offset-resolution
algorithms; it is not a replacement tzdata distribution. PEP 495 `fold`
semantics are also outside pytz's API and are not added here. Batch timestamps
must fit signed `int64` POSIX seconds; `datetime64` inputs containing `NaT` or
subsecond values are rejected instead of being silently narrowed.

## Install and verify

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` produces `dist/libmojo-pytz.so`. The Python binding also
rebuilds a missing or stale library on first use. Set `MOJOPYTZ_LIB` to load a
prebuilt shared library from another location.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. These are best-of-three wall-clock times and include NumPy output
allocation. The two UTC rows compare against a Python `bisect_right` loop over
the exact same integer transition table; the localization row compares the
object-producing APIs `localize_many(..., is_dst=False)` and
`pytz.timezone(...).localize(..., is_dst=False)`.

| case | mojo-pytz | pytz reference | result |
| --- | ---: | ---: | ---: |
| UTC offsets, sorted (250k) | 1.72 ms | 100.37 ms | 58.26x faster |
| UTC offsets, shuffled (250k) | 10.28 ms | 113.52 ms | 11.04x faster |
| localize, ordinary times (100k) | 99.72 ms | 706.87 ms | 7.09x faster |

Sorted input is especially fast because the kernel walks the timestamps and
transition table together. Shuffled input uses an independent binary search
per timestamp. Full localization has a smaller gain because both sides must
still construct 100,000 Python `datetime` objects.

There is no GPU backend. These kernels are branch-heavy integer table searches,
while full localization is dominated by Python object construction. Device
transfers would add overhead to work with no dense arithmetic loop. Independent
large local-time batches run in parallel; smaller batches stay serial to avoid
thread-launch overhead.

## How it works

pytz timezone objects expose UTC transition datetimes and a parallel table of
offset, DST, and abbreviation records. `MojoTimezone` converts those once into
contiguous NumPy arrays:

- transition instants and offsets are signed 64-bit integer seconds;
- DST predicates and result statuses are signed 8-bit integers;
- selected period indices are signed 32-bit integers.

The arrays cross ctypes as integer addresses. Existing contiguous `int64` NumPy
inputs remain zero-copy. `src/capi.mojo` rebuilds the addresses as
`UnsafePointer[..., AnyOrigin[mut=True]]`, matching Mojo's non-parametric C ABI
requirements. One FFI call handles the whole batch and Mojo never allocates or
owns Python memory.

UTC resolution uses a linear merge for monotonic input and binary search
otherwise. Local resolution searches only periods whose offsets could map a
wall time back into their UTC interval. Zero matches identifies a forward gap,
two matches identifies a fold, and `is_dst` selects the same side pytz does.
The Python layer then maps the selected table rows back to pytz's fixed-offset
tzinfo instances.

## License

MIT
