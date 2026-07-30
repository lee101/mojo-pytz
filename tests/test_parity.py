"""Behavioral parity with pytz on scalar and bulk offset resolution."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta

import numpy as np
import pytest
import pytz

import mojopytz


def signature(value):
    return value.replace(tzinfo=None), value.utcoffset(), value.dst(), value.tzname()


def utc_seconds(value):
    delta = value - datetime(1970, 1, 1)
    return delta.days * 86400 + delta.seconds


def transition_cases(zone_name):
    zone = pytz.timezone(zone_name)
    transitions = getattr(zone, "_utc_transition_times", ())
    infos = getattr(zone, "_transition_info", ())
    cases = []
    for index in range(1, len(transitions)):
        transition = transitions[index]
        if transition.year not in (1915, 1945, 2000, 2020, 2037):
            continue
        for info in (infos[index - 1], infos[index]):
            edge = transition + info[0]
            cases.extend(
                [edge - timedelta(seconds=1), edge, edge + timedelta(seconds=1)]
            )
    return list(dict.fromkeys(cases))


def test_version_and_timezone_lists_match_upstream():
    assert mojopytz.VERSION == pytz.VERSION
    assert mojopytz.OLSON_VERSION == pytz.OLSON_VERSION
    assert mojopytz.all_timezones == pytz.all_timezones
    assert mojopytz.all_timezones_set == pytz.all_timezones_set
    assert mojopytz.common_timezones == pytz.common_timezones
    assert mojopytz.common_timezones_set == pytz.common_timezones_set


def test_country_mappings_match_upstream():
    assert mojopytz.country_names["NZ"] == pytz.country_names["NZ"]
    assert mojopytz.country_timezones["US"] == pytz.country_timezones["US"]


def test_exception_classes_are_upstream_classes():
    assert mojopytz.AmbiguousTimeError is pytz.AmbiguousTimeError
    assert mojopytz.InvalidTimeError is pytz.InvalidTimeError
    assert mojopytz.NonExistentTimeError is pytz.NonExistentTimeError
    assert mojopytz.UnknownTimeZoneError is pytz.UnknownTimeZoneError


def test_utc_and_fixed_offset_compatibility():
    assert mojopytz.UTC is pytz.UTC
    assert mojopytz.utc is pytz.utc
    assert mojopytz.timezone("UTC") is pytz.UTC
    assert mojopytz.FixedOffset(330) is pytz.FixedOffset(330)


def test_timezone_is_cached_and_unknown_zone_raises():
    assert mojopytz.timezone("US/Eastern") is mojopytz.timezone("US/Eastern")
    with pytest.raises(pytz.UnknownTimeZoneError):
        mojopytz.timezone("Mars/Olympus_Mons")


@pytest.mark.parametrize(
    "zone_name", ["US/Eastern", "Europe/London", "Asia/Kathmandu", "Etc/GMT+5"]
)
def test_scalar_localize_matches_pytz(zone_name):
    ours = mojopytz.timezone(zone_name)
    theirs = pytz.timezone(zone_name)
    for value in (datetime(1985, 1, 1), datetime(2020, 7, 15, 12, 34, 56)):
        assert signature(ours.localize(value)) == signature(theirs.localize(value))


def test_scalar_fromutc_matches_pytz():
    ours = mojopytz.timezone("Australia/Lord_Howe")
    theirs = pytz.timezone("Australia/Lord_Howe")
    for value in (datetime(1985, 1, 1), datetime(2020, 4, 4, 15, 30)):
        got = ours.fromutc(value.replace(tzinfo=ours))
        ref = theirs.fromutc(value.replace(tzinfo=theirs))
        assert signature(got) == signature(ref)


def test_scalar_normalize_matches_pytz():
    ours = mojopytz.timezone("US/Eastern")
    theirs = pytz.timezone("US/Eastern")
    start = datetime(2020, 3, 8, 0, 30)
    got = ours.normalize(ours.localize(start) + timedelta(hours=3))
    ref = theirs.normalize(theirs.localize(start) + timedelta(hours=3))
    assert signature(got) == signature(ref)


def test_direct_tzinfo_attachment_matches_pytz():
    ours = mojopytz.timezone("US/Eastern")
    theirs = pytz.timezone("US/Eastern")
    got = datetime(2020, 1, 1, tzinfo=ours)
    ref = datetime(2020, 1, 1, tzinfo=theirs)
    assert got.utcoffset() == ref.utcoffset()
    assert got.dst() == ref.dst()
    assert got.tzname() == ref.tzname()


@pytest.mark.parametrize(
    "zone_name",
    [
        "US/Eastern",
        "Europe/London",
        "Australia/Lord_Howe",
        "Pacific/Chatham",
        "Africa/Casablanca",
        "Europe/Dublin",
    ],
)
def test_transition_indices_match_pytz_bisect(zone_name):
    ours = mojopytz.timezone(zone_name)
    theirs = pytz.timezone(zone_name)
    transition_values = np.array(
        [utc_seconds(value) for value in theirs._utc_transition_times], dtype=np.int64
    )
    samples = np.unique(
        np.concatenate(
            [transition_values, transition_values - 1, transition_values + 1]
        )
    )
    ref = np.array(
        [
            max(0, bisect_right(transition_values, value) - 1)
            for value in samples
        ],
        dtype=np.int32,
    )
    assert np.array_equal(ours.transition_indices(samples), ref)
    assert np.array_equal(ours.transition_indices(samples[::-1]), ref[::-1])


@pytest.mark.parametrize(
    "zone_name",
    ["US/Eastern", "Europe/Amsterdam", "Asia/Kathmandu", "Pacific/Chatham"],
)
def test_offsets_dst_and_names_match_public_pytz(zone_name):
    ours = mojopytz.timezone(zone_name)
    theirs = pytz.timezone(zone_name)
    seconds = np.array(
        [-2_000_000_000, -1, 0, 500_000_000, 1_000_000_000, 1_600_000_000],
        dtype=np.int64,
    )
    refs = [
        theirs.fromutc(
            (datetime(1970, 1, 1) + timedelta(seconds=int(value))).replace(
                tzinfo=theirs
            )
        )
        for value in seconds
    ]
    assert np.array_equal(
        ours.offsets_at(seconds),
        [int(value.utcoffset().total_seconds()) for value in refs],
    )
    assert np.array_equal(
        ours.dst_at(seconds), [int(value.dst().total_seconds()) for value in refs]
    )
    assert np.array_equal(ours.tzname_at(seconds), [value.tzname() for value in refs])


def test_datetime64_input_preserves_shape():
    ours = mojopytz.timezone("US/Eastern")
    values = np.array(
        [["2020-01-01T00:00:00", "2020-07-01T00:00:00"]],
        dtype="datetime64[s]",
    )
    offsets = ours.offsets_at(values)
    assert offsets.shape == values.shape
    assert offsets.tolist() == [[-18000, -14400]]


def test_datetime64_requires_exact_seconds_and_rejects_nat():
    zone = mojopytz.timezone("Etc/GMT")
    with pytest.raises(ValueError, match="whole-second"):
        zone.offsets_at(np.array(["2020-01-01T00:00:00.001"], dtype="datetime64[ms]"))
    with pytest.raises(ValueError, match="NaT"):
        zone.offsets_at(np.array(["NaT"], dtype="datetime64[s]"))


def test_integer_timestamp_does_not_silently_narrow():
    zone = mojopytz.timezone("Etc/GMT")
    maximum = np.array([np.iinfo(np.int64).max], dtype=np.uint64)
    assert zone.offsets_at(maximum).tolist() == [0]
    overflow = np.array([np.uint64(np.iinfo(np.int64).max) + np.uint64(1)])
    with pytest.raises(OverflowError, match="signed int64"):
        zone.offsets_at(overflow)


def test_fromutc_many_matches_public_pytz():
    ours = mojopytz.timezone("US/Eastern")
    theirs = pytz.timezone("US/Eastern")
    values = [
        datetime(1969, 12, 31, 23, 59, 59),
        datetime(2020, 3, 8, 6, 59, 59),
        datetime(2020, 3, 8, 7),
        datetime(2020, 11, 1, 6),
    ]
    got = ours.fromutc_many(values)
    ref = [theirs.fromutc(value.replace(tzinfo=theirs)) for value in values]
    assert [signature(value) for value in got] == [signature(value) for value in ref]


@pytest.mark.parametrize(
    "zone_name",
    [
        "US/Eastern",
        "Europe/London",
        "Australia/Lord_Howe",
        "Pacific/Chatham",
        "Africa/Casablanca",
        "Europe/Warsaw",
        "Europe/Dublin",
    ],
)
@pytest.mark.parametrize("is_dst", [False, True])
def test_localize_many_transition_boundaries(zone_name, is_dst):
    values = transition_cases(zone_name)
    ours = mojopytz.timezone(zone_name)
    theirs = pytz.timezone(zone_name)
    got = ours.localize_many(values, is_dst=is_dst)
    ref = [theirs.localize(value, is_dst=is_dst) for value in values]
    assert [signature(value) for value in got] == [signature(value) for value in ref]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (datetime(2020, 11, 1, 1, 30), pytz.AmbiguousTimeError),
        (datetime(2020, 3, 8, 2, 30), pytz.NonExistentTimeError),
    ],
)
def test_localize_many_is_dst_none_raises_same_error(value, error):
    ours = mojopytz.timezone("US/Eastern")
    theirs = pytz.timezone("US/Eastern")
    with pytest.raises(error):
        ours.localize_many([value], is_dst=None)
    with pytest.raises(error):
        theirs.localize(value, is_dst=None)


def test_localize_many_rejects_aware_datetime():
    zone = mojopytz.timezone("US/Eastern")
    aware = zone.localize(datetime(2020, 1, 1))
    with pytest.raises(ValueError, match="Not naive datetime"):
        zone.localize_many([aware])


@pytest.mark.parametrize("count", [99_999, 100_000])
def test_localize_many_parallel_threshold(count):
    zone = mojopytz.timezone("US/Eastern")
    values = np.full(count, 1_577_836_800, dtype=np.int64)
    localized = zone.localize_many(values, is_dst=False)
    assert localized.shape == values.shape
    assert signature(localized[0]) == signature(
        zone._inner.localize(datetime(2020, 1, 1), is_dst=False)
    )
    assert signature(localized[-1]) == signature(localized[0])


def test_empty_batches_preserve_shape():
    zone = mojopytz.timezone("US/Eastern")
    empty = np.empty((2, 0), dtype=np.int64)
    assert zone.offsets_at(empty).shape == (2, 0)
    assert zone.localize_many(empty).shape == (2, 0)


def test_static_timezone_batch_methods():
    zone = mojopytz.timezone("Etc/GMT+5")
    values = np.array([-1_000_000_000, 0, 1_000_000_000], dtype=np.int64)
    assert zone.offsets_at(values).tolist() == [-18000, -18000, -18000]
    localized = zone.localize_many([datetime(1900, 1, 1), datetime(2100, 1, 1)])
    assert [value.utcoffset() for value in localized] == [
        timedelta(hours=-5),
        timedelta(hours=-5),
    ]


def test_invalid_bulk_input_type():
    zone = mojopytz.timezone("US/Eastern")
    with pytest.raises(TypeError, match="timestamps must be"):
        zone.offsets_at(["not a datetime"])
