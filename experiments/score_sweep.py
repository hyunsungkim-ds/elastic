"""One-at-a-time sweep over the scoring function bounds and the alignment penalties (paper Table 5).

Varies the clipping bound of each per-feature scoring function and each NW penalty while the
others stay at their default, then measures the accuracy of ELASTIC-NW per event category.

Input: the benchmark data cached by evaluate.load_data.
Output: experiments/results/score_sweep_{timestamp}.csv, one row per (config, match, category).

Run from the repo root:
    python experiments/score_sweep.py
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
    FPS,
    RESULT_DIR,
    aggregate_sync_accuracy,
    collect_rows,
    compute_sync_accuracy,
    load_data,
)
from sync import elastic_nw as enw
from sync import utils
from sync.elastic_nw import ELASTIC_NW
from sync.utils import linear_scoring_func

# Swept globals of each module; PBD and OD share `player_dist_func`, so they move together.
UTILS_KNOBS = [
    "player_dist_func",
    "ball_accel_func",
    "kick_dist_func",
    "positive_slope_penalty",
    "negative_slope_penalty",
]
ENW_KNOBS = [
    "GAP_EVENT",
    "GAP_FRAME",
    "REPEAT_PENALTY",
    "REJECT_THRESHOLD",
]
UTILS_DEFAULTS = {name: getattr(utils, name) for name in UTILS_KNOBS}
ENW_DEFAULTS = {name: getattr(enw, name) for name in ENW_KNOBS}


def _score_bound(knob: str, hi: float, increasing: bool) -> dict:
    """One scoring function clipped at `hi`."""
    return {knob: linear_scoring_func(0, hi, increasing=increasing)}


def _slope_bounds(v_ms: float) -> dict:
    """Both slope penalties clipped at the m/frame equivalent of `v_ms`."""
    s = v_ms / FPS
    return {
        "positive_slope_penalty": linear_scoring_func(0, s, increasing=False),
        "negative_slope_penalty": linear_scoring_func(-s, 0, increasing=True),
    }


# Sweep axes as (group, value, {knob: value}); the rest stay at their defaults.
CONFIGS: list[tuple[str, object, dict]] = [
    ("baseline", "default", {}),
    *[("ba_max", v, _score_bound("ball_accel_func", v, True)) for v in [10, 20, 40, 50]],  # default 30 (m/s2)
    *[("pbd_od_max", v, _score_bound("player_dist_func", v, False)) for v in [1, 5, 7, 9]],  # default 3 (m)
    *[("kd_max", v, _score_bound("kick_dist_func", v, True)) for v in [1, 5, 7, 9]],  # default 3 (m)
    *[("pbds_max", v, _slope_bounds(v)) for v in [1, 3, 5, 9]],  # default 7 (m/s)
    *[("gap_frame", v, {"GAP_FRAME": v}) for v in [-0.3, -0.1, 0.1, 0.3]],  # default 0
    *[("gap_event", v, {"GAP_EVENT": v}) for v in [-0.3, -0.1, 0.1, 0.3]],  # default 0
    *[("repeat", v, {"REPEAT_PENALTY": v}) for v in [0, -0.2, -0.3, -0.4]],  # default -0.1
]


def reset_defaults() -> None:
    """Restore every swept global to its import-time default."""
    for name, val in UTILS_DEFAULTS.items():
        setattr(utils, name, val)
    for name, val in ENW_DEFAULTS.items():
        setattr(enw, name, val)


# ---------------------------------------------------------------------------
# Sweep runners


def build_candidate_cache(cache: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Precompute candidate frames + kick dists + take-on features once per match.

    These depend only on geometry (never on the swept scoring/penalty knobs), so
    ELASTIC_NW.run() can skip recomputing them when `cand_frames` is injected. The Sportec
    benchmark has no take_on events, so nw_score_takeon is never invoked in this sweep.
    """
    reset_defaults()
    cand_cache: dict[str, pd.DataFrame] = {}
    for mid, data in cache.items():
        syncer = ELASTIC_NW(data["input_events"], data["tracking"])
        cands = syncer.find_candidate_frames()
        cands = syncer.calculate_kick_dists(cands)
        cands = syncer.calculate_takeon_features(cands)
        cand_cache[mid] = cands
    return cand_cache


def run_one_config(
    cache: dict[str, dict], cand_cache: dict[str, pd.DataFrame], overrides: dict
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Run ELASTIC_NW on each cached match under one config; "ALL" holds the aggregate."""
    reset_defaults()
    for name, val in overrides.items():
        setattr(utils if name in UTILS_DEFAULTS else enw, name, val)

    per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mid, data in cache.items():
        syncer = ELASTIC_NW(data["input_events"], data["tracking"])
        syncer.cand_frames = cand_cache[mid].copy()
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
    print("building candidate cache ...")
    cand_cache = build_candidate_cache(cache)

    all_rows: list[dict] = []
    baseline_table = None
    try:
        for group, value, overrides in CONFIGS:
            print("\n" + "=" * 72 + f"\n[{group}={value}]\n" + "=" * 72)
            result = run_one_config(cache, cand_cache, overrides)
            print(result["ALL"][1].round(4).to_string())
            all_rows.extend(collect_rows(group, value, result))
            if group == "baseline":
                baseline_table = result["ALL"][1].copy()

        # State-leak guard, kept only here: this is the one sweep that swaps out callables,
        # where a leaked override would silently shift every later config.
        recheck = run_one_config(cache, cand_cache, {})
        if baseline_table is not None and not recheck["ALL"][1].round(6).equals(baseline_table.round(6)):
            print("\nWARNING: trailing baseline differs from initial baseline (state leak?)")
        else:
            print("\nbaseline re-check OK (no state leak)")
    finally:
        reset_defaults()

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(RESULT_DIR) / f"score_sweep_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(f"total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
