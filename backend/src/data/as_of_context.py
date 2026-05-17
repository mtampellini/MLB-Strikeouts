"""AsOfContext: leakage should be impossible by construction.

Every data client inherits :class:`AsOfClient` and implements
``_fetch(cutoff_date, **kwargs)``. The public :meth:`AsOfClient.fetch` wrapper
validates the cutoff and walks the returned payload looking for any date or
ISO-8601 timestamp newer than cutoff. Any such record raises
:class:`LeakageError`.

Three guardrails:

- ``cutoff_date=None`` is the only way to skip the runtime check, and it logs
  a loud warning. This is "live mode" — fine for the daily pipeline, but
  tests should always pass a real cutoff_date.
- ``cutoff_date`` in the future raises ``ValueError`` — there is no legitimate
  use case for fetching tomorrow's data.
- The runtime data-walk can be disabled in production for performance by
  setting ``ASOF_DISABLE_RUNTIME_CHECK=1`` in the environment. It remains
  enabled in tests by default.

The walker is conservative: it tries to ISO-parse any string that looks like
``YYYY-MM-DD…`` and silently skips strings that don't parse. False positives
(strings that look like dates but aren't) would surface as loud LeakageErrors
and are easy to fix; false negatives (actual dates we missed) are the failure
mode we cannot tolerate, so we prefer the conservative side.
"""
from __future__ import annotations

import abc
import logging
import os
import warnings
from datetime import date, datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_RUNTIME_CHECK_ENV = "ASOF_DISABLE_RUNTIME_CHECK"


class LeakageError(Exception):
    """A client returned data timestamped after its cutoff_date."""


def runtime_check_enabled() -> bool:
    return os.environ.get(_RUNTIME_CHECK_ENV, "").lower() not in ("1", "true", "yes")


class AsOfClient(abc.ABC):
    """Base class for any data source that must respect an as-of date.

    Subclasses implement :meth:`_fetch`. Callers always use :meth:`fetch`.
    """

    @abc.abstractmethod
    def _fetch(self, cutoff_date: date | None, **kwargs: Any) -> Any:
        """Subclass implementation. Must return its data without leakage."""

    def fetch(self, *, cutoff_date: date | None, **kwargs: Any) -> Any:
        if cutoff_date is None:
            warnings.warn(
                f"{type(self).__name__}: LIVE MODE (cutoff_date=None) — "
                "leakage check disabled. Do not use for backtests or training.",
                stacklevel=2,
            )
            logger.warning(
                "%s: LIVE MODE (cutoff_date=None)", type(self).__name__
            )
        else:
            self._validate_cutoff(cutoff_date)

        result = self._fetch(cutoff_date, **kwargs)

        if cutoff_date is not None and runtime_check_enabled():
            self._check_no_leakage(result, cutoff_date)

        return result

    def _validate_cutoff(self, cutoff_date: date) -> None:
        if not isinstance(cutoff_date, date):
            raise TypeError(
                f"cutoff_date must be a datetime.date, got {type(cutoff_date).__name__}"
            )
        if isinstance(cutoff_date, datetime):
            cutoff_date = cutoff_date.date()
        if cutoff_date > date.today():
            raise ValueError(
                f"cutoff_date {cutoff_date} is in the future (today={date.today()})"
            )

    def _check_no_leakage(self, result: Any, cutoff_date: date) -> None:
        for ts in self._iter_dates(result):
            if ts > cutoff_date:
                raise LeakageError(
                    f"{type(self).__name__}: record dated {ts} exceeds "
                    f"cutoff {cutoff_date}"
                )

    def _iter_dates(self, obj: Any) -> Iterable[date]:
        """Yield every date/datetime found inside obj (recursively)."""
        if isinstance(obj, datetime):
            if obj.tzinfo is not None:
                yield obj.astimezone(timezone.utc).date()
            else:
                yield obj.date()
            return
        if isinstance(obj, date):
            yield obj
            return
        if isinstance(obj, str):
            yield from _maybe_parse_iso_date(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                yield from self._iter_dates(v)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                yield from self._iter_dates(item)
            return


def _maybe_parse_iso_date(s: str) -> Iterable[date]:
    """Try to ISO-parse strings that look like dates; silently skip others."""
    if len(s) < 10 or s[4] != "-" or s[7] != "-":
        return
    head = s[:10]
    if not (head[:4].isdigit() and head[5:7].isdigit() and head[8:10].isdigit()):
        return
    try:
        normalized = s.replace("Z", "+00:00")
        if len(s) == 10:
            yield date.fromisoformat(head)
        else:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is not None:
                yield dt.astimezone(timezone.utc).date()
            else:
                yield dt.date()
    except ValueError:
        return
