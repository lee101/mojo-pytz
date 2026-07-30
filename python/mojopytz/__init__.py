"""pytz-compatible timezone objects with Mojo batch offset resolution."""

from __future__ import annotations

import pytz as _pytz

from ._lib import build
from .timezone import MojoTimezone, timezone

AmbiguousTimeError = _pytz.AmbiguousTimeError
InvalidTimeError = _pytz.InvalidTimeError
NonExistentTimeError = _pytz.NonExistentTimeError
UnknownTimeZoneError = _pytz.UnknownTimeZoneError
FixedOffset = _pytz.FixedOffset
UTC = _pytz.UTC
utc = _pytz.utc

all_timezones = _pytz.all_timezones
all_timezones_set = _pytz.all_timezones_set
common_timezones = _pytz.common_timezones
common_timezones_set = _pytz.common_timezones_set
country_names = _pytz.country_names
country_timezones = _pytz.country_timezones

OLSON_VERSION = _pytz.OLSON_VERSION
VERSION = _pytz.VERSION
__version__ = "0.1.0"

__all__ = [
    "timezone",
    "MojoTimezone",
    "UTC",
    "utc",
    "FixedOffset",
    "AmbiguousTimeError",
    "InvalidTimeError",
    "NonExistentTimeError",
    "UnknownTimeZoneError",
    "all_timezones",
    "all_timezones_set",
    "common_timezones",
    "common_timezones_set",
    "country_names",
    "country_timezones",
    "OLSON_VERSION",
    "VERSION",
    "build",
]
