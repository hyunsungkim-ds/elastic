"""One-at-a-time sweep over the per-feature weights of the pairwise scoring (paper Table 4).

Varies each weight of nw_score_major / nw_score_minor while the others stay at their default,
then measures the accuracy of ELASTIC-NW per event category.

Input: the benchmark data cached by evaluate.load_data.
Output: experiments/results/coeff_sweep_{timestamp}.csv, one row per (weight, value, match, category).

Run from the repo root:
    python experiments/coeff_sweep.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments.evaluate import (
    RESULT_DIR,
    aggregate_sync_accuracy,
    collect_rows,
    compute_sync_accuracy,
    load_data,
)
from sync import utils
from sync.elastic_nw import ELASTIC_NW

# Each sweep group maps to one weight attribute on `sync.utils`. Comment out a group to skip it.
GROUP_TO_ATTR: dict[str, str] = {
    "BA": "BA_WEIGHT",  # ball_accel
    "PBD": "PBD_WEIGHT",  # player_dist
    "KD": "KD_WEIGHT",  # kick_dist
    "PBDS": "PBDS_WEIGHT",  # player_dist_slope (major events only)
    "OD": "OD_WEIGHT",  # oppo_dist (minor events only)
}
UTILS_DEFAULTS = {name: getattr(utils, name) for name in GROUP_TO_ATTR.values()}

# Sweep axes as (group, value, {attr: value}); the rest stay at their defaults.
SWEEP_VALUES = [0.00, 0.50]  # the default weight is covered by the baseline config
CONFIGS: list[tuple[str, object, dict]] = [
    ("baseline", "default", {}),
    *[(group, v, {attr: v}) for group, attr in GROUP_TO_ATTR.items() for v in SWEEP_VALUES],
]


def reset_defaults() -> None:
    """Restore every swept global to its import-time default."""
    for name, val in UTILS_DEFAULTS.items():
        setattr(utils, name, val)


def run_one_config(cache: dict[str, dict], overrides: dict) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Run ELASTIC_NW on each cached match under one config; "ALL" holds the aggregate."""
    reset_defaults()
    for name, val in overrides.items():
        setattr(utils, name, val)

    per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mid, data in cache.items():
        syncer = ELASTIC_NW(data["input_events"], data["tracking"])
        synced = syncer.run()
        counts, rates, _ = compute_sync_accuracy(synced, data["gt"])
        per_match[mid] = (counts, rates)

    agg_counts, agg_rates = aggregate_sync_accuracy(per_match)
    per_match["ALL"] = (agg_counts, agg_rates)
    return per_match


def main() -> None:
    t0 = time.time()
    print("loading data ...")
    cache = load_data()

    all_rows: list[dict] = []
    try:
        for group, value, overrides in CONFIGS:
            print("\n" + "=" * 72 + f"\n[{group}={value}]\n" + "=" * 72)
            result = run_one_config(cache, overrides)
            print(result["ALL"][1].round(4).to_string())
            all_rows.extend(collect_rows(group, value, result))
    finally:
        reset_defaults()

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(RESULT_DIR) / f"coeff_sweep_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(f"total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
