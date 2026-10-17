"""Bottom-of-terminal progress via tqdm.auto (clears when finished)."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Any, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

# Keep desc / postfix constant width so dynamic_ncols leaves {bar} stable.
_DEFAULT_DESC_WIDTH = 36
_DEFAULT_POSTFIX_WIDTH = 48


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return text
    if len(text) > width:
        if width == 1:
            return "…"
        return text[: width - 1] + "…"
    return text.ljust(width)


def _bar_format(total: int | None) -> str:
    """Fixed-width n/total fields sized to ``total`` so the bar does not jitter."""
    digits = max(len(str(max(int(total or 0), 0))), 4)
    return (
        "{desc}: {percentage:3.0f}%|{bar}| "
        f"{{n:{digits}d}}/{{total:{digits}d}} "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )


def progress(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    **kwargs: object,
) -> tqdm:
    """
    tqdm.auto bar pinned to the bottom row.

    - leave=False: disappears when closed
    - shows desc, rate, and ETA by default
    """
    options: dict[str, object] = {
        "leave": False,
        "desc": "progress",
        "bar_format": _bar_format(total),
        "dynamic_ncols": True,
        "file": sys.stderr,
        "position": 0,
        "mininterval": 0.2,
        "smoothing": 0.05,
    }
    options.update(kwargs)
    if iterable is None:
        return tqdm(total=total, **options)
    return tqdm(iterable, total=total, **options)


class ProgressBar:
    """Context-manager wrapper around ``progress`` for manual ``update`` loops."""

    def __init__(
        self,
        total: int,
        *,
        desc_width: int = _DEFAULT_DESC_WIDTH,
        postfix_width: int = _DEFAULT_POSTFIX_WIDTH,
        **kwargs: object,
    ) -> None:
        self._desc_width = int(desc_width)
        self._postfix_width = int(postfix_width)
        total_i = max(int(total), 0)
        if "desc" in kwargs and kwargs["desc"] is not None:
            kwargs = {**kwargs, "desc": _fit(str(kwargs["desc"]), self._desc_width)}
        kwargs.setdefault("bar_format", _bar_format(total_i))
        self._pbar = progress(total=total_i, **kwargs)
        # Reserve postfix width immediately so the bar does not shrink on first update.
        self._pbar.set_postfix_str(f" {_fit('', self._postfix_width)}", refresh=False)

    def update(self, n: int = 1) -> None:
        self._pbar.update(int(n))

    def set_description(self, desc: str | None) -> None:
        # set_description_str: do not append an extra ": " (bar_format already has one).
        text = _fit("" if desc is None else str(desc), self._desc_width)
        self._pbar.set_description_str(text, refresh=False)

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, **kwargs: Any) -> None:
        data: dict[str, Any] = {}
        if ordered_dict is not None:
            data.update(ordered_dict)
        data.update(kwargs)
        parts = [f"{key}={value}" for key, value in data.items()]
        text = _fit(", ".join(parts), self._postfix_width)
        # Leading space matches tqdm's default postfix separator look.
        self._pbar.set_postfix_str(f" {text}", refresh=False)

    def close(self) -> None:
        self._pbar.close()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.close()
        return False


def track(iterable: Iterable[T], *, total: int | None = None, **kwargs: object) -> Iterator[T]:
    """Iterate with a disappearing bottom progress bar."""
    return iter(progress(iterable, total=total, **kwargs))
