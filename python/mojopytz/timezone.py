"""pytz-compatible timezone objects with Mojo batch resolution methods."""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta, tzinfo
from functools import lru_cache
from typing import Any

import numpy as np
import pytz as _pytz

from ._lib import addr, lib

_EPOCH = datetime(1970, 1, 1)


def _datetime_seconds(value: datetime) -> int:
    naive = value if value.tzinfo is None else value.replace(tzinfo=None)
    delta = naive - _EPOCH
    return delta.days * 86400 + delta.seconds


def _object_seconds(
    objects: list[datetime],
    reject_aware: bool,
) -> np.ndarray:
    seconds = np.empty(len(objects), dtype=np.int64)
    for index, value in enumerate(objects):
        if not isinstance(value, datetime):
            raise TypeError(
                "timestamps must be datetimes, datetime64 values, or integer seconds"
            )
        if reject_aware and value.tzinfo is not None:
            raise ValueError("Not naive datetime (tzinfo is already set)")
        seconds[index] = _datetime_seconds(value)
    return seconds


def _seconds_array(
    values: Any,
    *,
    reject_aware: bool = False,
) -> tuple[np.ndarray, tuple[int, ...], list[datetime] | None]:
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.datetime64):
        shape = values.shape
        if np.isnat(values).any():
            raise ValueError("NaT timestamps are not supported")
        second_values = values.astype("datetime64[s]")
        if not np.array_equal(values, second_values.astype(values.dtype)):
            raise ValueError("datetime64 timestamps must have whole-second precision")
        seconds = np.ascontiguousarray(second_values.astype(np.int64).reshape(-1))
        return seconds, shape, None

    if isinstance(values, list) and (
        not values or isinstance(values[0], datetime)
    ):
        return _object_seconds(values, reject_aware), (len(values),), values

    if isinstance(values, tuple) and (
        not values or isinstance(values[0], datetime)
    ):
        objects = list(values)
        return _object_seconds(objects, reject_aware), (len(objects),), objects

    array = np.asarray(values)
    shape = array.shape
    if np.issubdtype(array.dtype, np.integer):
        i64 = np.iinfo(np.int64)
        if array.size and (np.any(array < i64.min) or np.any(array > i64.max)):
            raise OverflowError("integer timestamps must fit in signed int64")
        return np.ascontiguousarray(array, dtype=np.int64).reshape(-1), shape, None

    objects = array.reshape(-1).tolist()
    return _object_seconds(objects, reject_aware), shape, objects


def _transition_seconds(value: datetime) -> int:
    return calendar.timegm(value.utctimetuple())


class MojoTimezone(tzinfo):
    """A pytz timezone with scalar compatibility and compiled batch lookup."""

    def __init__(self, inner: tzinfo):
        self._inner = inner
        self.zone = getattr(inner, "zone", str(inner))

        transition_datetimes = getattr(inner, "_utc_transition_times", None)
        transition_info = getattr(inner, "_transition_info", None)
        if transition_datetimes is None:
            offset = inner.utcoffset(datetime(2000, 1, 1))
            dst = inner.dst(datetime(2000, 1, 1))
            name = inner.tzname(datetime(2000, 1, 1))
            transition_datetimes = [datetime.min]
            transition_info = [(offset, dst, name)]

        self._transition_datetimes = tuple(transition_datetimes)
        self._transition_info = tuple(transition_info)
        self._transitions = np.ascontiguousarray(
            [_transition_seconds(value) for value in transition_datetimes], dtype=np.int64
        )
        self._offsets = np.ascontiguousarray(
            [int(info[0].total_seconds()) for info in transition_info], dtype=np.int64
        )
        self._dst_seconds = np.ascontiguousarray(
            [int(info[1].total_seconds()) for info in transition_info], dtype=np.int64
        )
        self._dst_flags = np.ascontiguousarray(self._dst_seconds != 0, dtype=np.int8)
        self._names = np.asarray([info[2] for info in transition_info], dtype=object)
        self._max_offset = int(max(86400, np.max(np.abs(self._offsets))))
        tzinfos = getattr(self._inner, "_tzinfos", None)
        self._fixed_tzinfos = tuple(
            tzinfos[info] if tzinfos is not None else self._inner
            for info in self._transition_info
        )

    def __repr__(self) -> str:
        return repr(self._inner)

    def __str__(self) -> str:
        return str(self._inner)

    def _for_inner(self, dt: datetime | None) -> datetime | None:
        if dt is not None and dt.tzinfo is self:
            return dt.replace(tzinfo=self._inner)
        return dt

    def utcoffset(self, dt: datetime | None, is_dst: bool | None = None):
        return self._inner.utcoffset(self._for_inner(dt), is_dst)

    def dst(self, dt: datetime | None, is_dst: bool | None = None):
        return self._inner.dst(self._for_inner(dt), is_dst)

    def tzname(self, dt: datetime | None, is_dst: bool | None = None):
        return self._inner.tzname(self._for_inner(dt), is_dst)

    def fromutc(self, dt: datetime) -> datetime:
        return self._inner.fromutc(self._for_inner(dt))

    def localize(self, dt: datetime, is_dst: bool | None = False) -> datetime:
        return self._inner.localize(dt, is_dst)

    def normalize(self, dt: datetime) -> datetime:
        return self._inner.normalize(self._for_inner(dt))

    def _utc_indices(self, seconds: np.ndarray) -> np.ndarray:
        indices = np.empty(seconds.size, dtype=np.int32)
        if seconds.size:
            lib().mptz_resolve_utc(
                addr(self._transitions, np.int64),
                self._transitions.size,
                addr(seconds, np.int64),
                seconds.size,
                addr(indices, np.int32, writable=True),
            )
        return indices

    def transition_indices(self, utc_timestamps: Any) -> np.ndarray:
        """Return pytz transition-table indices for UTC-axis timestamps."""
        seconds, shape, _ = _seconds_array(utc_timestamps)
        return self._utc_indices(seconds).reshape(shape)

    def offsets_at(self, utc_timestamps: Any) -> np.ndarray:
        """Return UTC offsets in integer seconds for UTC-axis timestamps."""
        seconds, shape, _ = _seconds_array(utc_timestamps)
        return self._offsets[self._utc_indices(seconds)].reshape(shape)

    def dst_at(self, utc_timestamps: Any) -> np.ndarray:
        """Return DST adjustments in integer seconds for UTC-axis timestamps."""
        seconds, shape, _ = _seconds_array(utc_timestamps)
        return self._dst_seconds[self._utc_indices(seconds)].reshape(shape)

    def tzname_at(self, utc_timestamps: Any) -> np.ndarray:
        """Return timezone abbreviations for UTC-axis timestamps."""
        seconds, shape, _ = _seconds_array(utc_timestamps)
        return self._names[self._utc_indices(seconds)].reshape(shape)

    def fromutc_many(self, datetimes: Any) -> np.ndarray:
        """Vectorized form of ``fromutc`` for naïve UTC wall-clock datetimes."""
        seconds, shape, objects = _seconds_array(datetimes)
        indices = self._utc_indices(seconds)
        if objects is None:
            objects = [
                _EPOCH + timedelta(seconds=int(value))
                for value in seconds
            ]

        result = np.empty(len(objects), dtype=object)
        for pos, (value, index) in enumerate(zip(objects, indices)):
            naive = value.replace(tzinfo=None)
            info = self._transition_info[int(index)]
            fixed = self._fixed_tzinfos[int(index)]
            result[pos] = (naive + info[0]).replace(tzinfo=fixed)
        return result.reshape(shape)

    def localize_many(
        self, datetimes: Any, is_dst: bool | None = False
    ) -> np.ndarray:
        """Vectorized form of ``localize``, including pytz fold/gap semantics."""
        seconds, shape, objects = _seconds_array(datetimes, reject_aware=True)
        if objects is None:
            objects = [_EPOCH + timedelta(seconds=int(value)) for value in seconds]

        indices = np.empty(seconds.size, dtype=np.int32)
        statuses = np.empty(seconds.size, dtype=np.int8)
        flag = -1 if is_dst is None else int(bool(is_dst))
        if seconds.size:
            lib().mptz_resolve_local(
                addr(self._transitions, np.int64),
                addr(self._offsets, np.int64),
                addr(self._dst_flags, np.int8),
                self._transitions.size,
                addr(seconds, np.int64),
                seconds.size,
                flag,
                self._max_offset,
                addr(indices, np.int32, writable=True),
                addr(statuses, np.int8, writable=True),
            )

        bad = np.flatnonzero(statuses) if seconds.size else np.empty(0, dtype=int)
        if bad.size:
            pos = int(bad[0])
            if statuses[pos] == 1:
                raise _pytz.NonExistentTimeError(objects[pos])
            raise _pytz.AmbiguousTimeError(objects[pos])

        result = np.empty(len(objects), dtype=object)
        for pos, (value, index) in enumerate(zip(objects, indices)):
            result[pos] = value.replace(tzinfo=self._fixed_tzinfos[int(index)])
        return result.reshape(shape)


@lru_cache(maxsize=None)
def timezone(zone: str) -> MojoTimezone | tzinfo:
    """Return a cached timezone, matching ``pytz.timezone(zone)``."""
    inner = _pytz.timezone(zone)
    if inner is _pytz.UTC:
        return _pytz.UTC
    return MojoTimezone(inner)
