"""Load market-feature specs from config/feature.yaml."""

from __future__ import annotations

from pathlib import Path

from util.yaml_io import load_yaml

FEATURE_FUNCTIONS = ("pct_change", "rolling_std", "ma_gap")


def load_feature_specs(path: Path | str) -> list[dict[str, str | int]]:
    """Return feature specs in file order: column, fn, window."""
    raw = load_yaml(path)
    if not raw:
        raise ValueError(f"feature file is empty: {path}")

    specs: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for column, spec in raw.items():
        name = str(column)
        if name in seen:
            raise ValueError(f"feature column {name!r} is listed more than once")
        if not isinstance(spec, dict):
            raise TypeError(f"feature {name!r} must be a mapping with fn and window")
        if "fn" not in spec or "window" not in spec:
            raise ValueError(f"feature {name!r} needs fn and window")
        fn = str(spec["fn"])
        if fn not in FEATURE_FUNCTIONS:
            raise ValueError(
                f"feature {name!r} fn {fn!r} is unknown. Choices: {list(FEATURE_FUNCTIONS)}"
            )
        window = int(spec["window"])
        if window < 1:
            raise ValueError(f"feature {name!r} window must be >= 1")
        seen.add(name)
        specs.append({"column": name, "fn": fn, "window": window})
    return specs
