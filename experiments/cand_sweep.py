"""One-at-a-time sweep over the candidate frame selection knobs of ELASTIC-NW (paper Table 3).

Measures the candidate coverage over the GT event frames (the W2 accuracy of an oracle that
always picks the right candidate, which upper-bounds any aligner), the end-to-end accuracy of
ELASTIC-NW, and the per-match runtime of each stage.

Input: the benchmark data cached by evaluate.load_data.
Output: experiments/results/cand_sweep_{timestamp}.csv, one row per (config, match, category).

Run from the repo root:
    python experiments/cand_sweep.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.evaluate import (
    EVENT_CATS,
    RESULT_DIR,
    accuracy_row,
    aggregate_sync_accuracy,
    compute_sync_accuracy,
    load_data,
)
from sync import config
from sync import elastic_nw as enw
from sync.elastic_nw import ELASTIC_NW

REPORT_BUFFER = 2  # coverage is reported at this tolerance only (W2, the paper's headline metric)
COV_COL = f"cov_within_{REPORT_BUFFER}"

# Swept globals of sync.elastic_nw.
ENW_KNOBS = [
    "CAND_USE_PBD_VALLEYS",
    "CAND_USE_BBD_VALLEYS",
    "CAND_USE_ACCEL_PEAKS",
    "CAND_PBD_MAX",
    "CAND_BBD_MAX",
    "CAND_BALL_HEIGHT_MAX",
]
ENW_DEFAULTS = {name: getattr(enw, name) for name in ENW_KNOBS}

# Sweep axes as (group, value, {knob: value}); the rest stay at their defaults.
CONFIGS: list[tuple[str, object, dict]] = [
    ("baseline", "default", {}),
    ("sources", "dist_only", {"CAND_USE_ACCEL_PEAKS": False}),  # default uses both sources
    ("sources", "accel_only", {"CAND_USE_PBD_VALLEYS": False, "CAND_USE_BBD_VALLEYS": False}),
    *[("pbd_max", v, {"CAND_PBD_MAX": v}) for v in [1.0, 2.0, 4.0, 5.0, np.inf]],  # default 3 (m)
    *[("height_max", v, {"CAND_BALL_HEIGHT_MAX": v}) for v in [1.0, 2.0, 3.0, 5.0, np.inf]],  # default 4 (m)
]


def reset_defaults() -> None:
    """Restore every swept global to its import-time default."""
    for name, val in ENW_DEFAULTS.items():
        setattr(enw, name, val)


# ---------------------------------------------------------------------------
# Candidate coverage over the GT event frames


def _nearest_dists(cands: pd.DataFrame, targets: pd.DataFrame, player_col: str, frame_col: str) -> np.ndarray:
    """|nearest same-player candidate frame - target frame| per target row (inf if none)."""
    frames_by_player = {p: np.sort(g["frame_id"].unique()) for p, g in cands.groupby("player_id")}
    out = np.full(len(targets), np.inf)
    for k, (pid, frame) in enumerate(zip(targets[player_col], targets[frame_col])):
        arr = frames_by_player.get(pid)
        if arr is None or len(arr) == 0 or pd.isna(frame):
            continue
        pos = np.searchsorted(arr, frame)
        best = np.inf
        if pos < len(arr):
            best = abs(arr[pos] - frame)
        if pos > 0:
            best = min(best, abs(arr[pos - 1] - frame))
        out[k] = best
    return out


def compute_coverage(cands: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Per-category counts of GT events with a same-player candidate within REPORT_BUFFER frames."""
    gt = gt.copy()
    gt["event_cat"] = gt["spadl_type"].map(config.EVENT_CAT_MAP)
    starts = gt[gt["event_cat"].notna()].reset_index(drop=True)
    start_dists = _nearest_dists(cands, starts, "player_id", "frame_id")

    kick_mask = starts["event_cat"].isin(["pass_like", "set_piece"]) & starts["receive_frame_id"].notna()
    ends = starts[kick_mask].reset_index(drop=True)
    end_dists = _nearest_dists(cands, ends, "receiver_id", "receive_frame_id")

    dists = {cat: start_dists[(starts["event_cat"] == cat).to_numpy()] for cat in EVENT_CATS}
    dists["event_start"] = start_dists
    dists["event_end"] = end_dists
    dists["total"] = np.concatenate([start_dists, end_dists])

    rows = {cat: {"n_events": len(d), COV_COL: int((d <= REPORT_BUFFER).sum())} for cat, d in dists.items()}
    counts = pd.DataFrame.from_dict(rows, orient="index")
    counts["n_candidates"] = len(cands)
    return counts


def aggregate_coverage(per_match: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-match coverage counts into a single table."""
    return sum(df for df in per_match.values())


def coverage_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """Turn coverage counts into the per-category coverage rate, keeping n_candidates."""
    rates = counts[["n_candidates"]].copy()
    rates[COV_COL] = counts[COV_COL] / counts["n_events"].replace(0, np.nan)
    return rates


# ---------------------------------------------------------------------------
# Sweep runners


def run_one_config(cache: dict[str, dict], overrides: dict) -> dict[str, dict]:
    """Run ELASTIC_NW on each cached match under one config; "ALL" holds the aggregate.

    Returns the coverage, accuracy, and runtime blocks, each keyed by match_id plus "ALL".
    """
    reset_defaults()
    for name, val in overrides.items():
        setattr(enw, name, val)

    cov_per_match: dict[str, pd.DataFrame] = {}
    acc_per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    time_per_match: dict[str, dict[str, float]] = {}
    for mid, data in cache.items():
        # Stage 1: candidate selection (detection + kick-dist/take-on features)
        t0 = time.time()
        syncer = ELASTIC_NW(data["input_events"], data["tracking"])
        cands = syncer.find_candidate_frames()
        cands = syncer.calculate_kick_dists(cands)
        cands = syncer.calculate_takeon_features(cands)
        t_cand = time.time() - t0

        # Stage 2: NW alignment + postprocessing
        syncer.cand_frames = cands
        t1 = time.time()
        synced = syncer.run()
        t_align = time.time() - t1

        time_per_match[mid] = {"cand_time": t_cand, "align_time": t_align}
        cov_per_match[mid] = compute_coverage(cands, data["gt"])
        counts, rates, _ = compute_sync_accuracy(synced, data["gt"])
        acc_per_match[mid] = (counts, rates)

    cov_per_match["ALL"] = aggregate_coverage(cov_per_match)
    agg_counts, agg_rates = aggregate_sync_accuracy(acc_per_match)
    acc_per_match["ALL"] = (agg_counts, agg_rates)

    time_per_match["ALL"] = pd.DataFrame(time_per_match).T.mean().to_dict()
    return {"cov": cov_per_match, "acc": acc_per_match, "time": time_per_match}


def collect_rows(sweep_group: str, target_value: object, result: dict[str, dict]) -> list[dict]:
    """One CSV row per (match, category): the sibling sweeps' accuracy columns plus coverage and runtime.

    Not `evaluate.collect_rows` itself because each row also carries the match's candidate
    count, coverage rate, and per-stage runtime (repeated across its category rows).
    """
    rows: list[dict] = []
    for mid, (counts, rates) in result["acc"].items():
        cov = coverage_rates(result["cov"][mid]).to_dict("index")  # keeps n_candidates an int
        for cat in rates.index:
            row = {"sweep_group": sweep_group, "target_value": target_value, "match_id": mid, "category": cat}
            row.update(cov[cat])
            row.update(accuracy_row(counts, rates, cat))
            row.update(result["time"][mid])
            rows.append(row)
    return rows


def main() -> None:
    t0 = time.time()
    print("loading data ...")
    cache = load_data()

    all_rows: list[dict] = []
    try:
        for group, value, overrides in CONFIGS:
            print("\n" + "=" * 72 + f"\n[{group}={value}]\n" + "=" * 72)
            result = run_one_config(cache, overrides)
            rows = collect_rows(group, value, result)
            all_rows.extend(rows)

            view = pd.DataFrame(rows)
            view = view[view["match_id"] == "ALL"].set_index("category")
            cols = ["n_candidates", COV_COL, f"within_{REPORT_BUFFER}", "valid", "mean_diff"]
            print(view[cols].round(4).to_string())
            t = result["time"]["ALL"]
            print(f"runtime (mean over matches): cand {t['cand_time']:.1f}s, align {t['align_time']:.1f}s")
    finally:
        reset_defaults()

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(RESULT_DIR) / f"cand_sweep_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(f"total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
