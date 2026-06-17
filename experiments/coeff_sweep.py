"""Sweep per-term weights of nw_score_major / nw_score_minor on Sportec 3 matches.

Edit the SETTINGS block below, then run from repo root:

    python experiments/coeff_sweep.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.evaluate import aggregate_sync_accuracy, compute_sync_accuracy
from sync import schema, utils
from sync.elastic_nw import ELASTIC_NW
from tools.sportec_data import SportecData

# ===========================================================
# ===== SETTINGS (edit this block to control the sweep) =====
# ===========================================================
SWEEP_VALUES = [0.00, 0.25, 0.50]  # values applied to each target coefficient
SWEEP_GROUPS = ["BA", "PBD", "KD", "PBDS", "OD"]  # comment out a group to skip it
MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY"]
INPUT_DIR = "data/sportec/event_corrected"  # source of {mid}.parquet (input events)
GT_DIR = "data/sportec/event_synced/gt"  # source of {mid}.parquet (ground truth)
RESULT_DIR = "experiments"
DETECT_CONTROLS = True
# ===========================================================


# Each sweep group maps to one attribute on `sync.utils` to override.
GROUP_TO_ATTR: dict[str, str] = {
    "BA": "BA_WEIGHT",  # ball_accel
    "PBD": "PBD_WEIGHT",  # player_dist
    "KD": "KD_WEIGHT",  # kick_dist
    "PBDS": "PBDS_WEIGHT",  # player_dist_slope
    "OD": "OD_WEIGHT",  # oppo_dist
}
BASELINE_WEIGHT = 0.25


def reset_weights() -> None:
    utils.BA_WEIGHT = BASELINE_WEIGHT
    utils.PBD_WEIGHT = BASELINE_WEIGHT
    utils.KD_WEIGHT = BASELINE_WEIGHT
    utils.PBDS_WEIGHT = BASELINE_WEIGHT
    utils.OD_WEIGHT = BASELINE_WEIGHT


def load_data() -> dict[str, dict]:
    """Load input_events / gt (from cached parquet) + tracking (via SportecData) once per match.

    Match the eval_sportec.ipynb flow: input events come from
    {INPUT_EVENTS_DIR}/{mid}.parquet and GT from {GT_DIR}/{mid}.parquet,
    NOT from build_corrected_events()/build_gt() — those rebuild from _merged.csv
    which lags behind the GT parquet that has been hand-edited.
    """
    cache: dict[str, dict] = {}
    elastic_cols = list(schema.elastic_event_schema.columns.keys())
    for mid in MATCH_IDS:
        input_events = pd.read_parquet(f"{INPUT_DIR}/{mid}.parquet")
        gt = pd.read_parquet(f"{GT_DIR}/{mid}.parquet")
        cache[mid] = {
            "input_events": input_events[elastic_cols],
            "gt": gt,
            "tracking": SportecData(mid).format_tracking_for_syncer(),
        }
    return cache


def run_one_config(cache: dict[str, dict], overrides: dict[str, float]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Run ELASTIC_NW on each cached match with the given weight overrides.

    Returns {match_id: (counts, rates), ..., "ALL": (counts_agg, rates_agg)}.
    """
    reset_weights()
    for attr, val in overrides.items():
        setattr(utils, attr, val)

    per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mid, data in cache.items():
        syncer = ELASTIC_NW(data["input_events"], data["tracking"], detect_controls=DETECT_CONTROLS)
        synced = syncer.run()
        counts, rates, _ = compute_sync_accuracy(synced, data["gt"])
        per_match[mid] = (counts, rates)

    agg_counts, agg_rates = aggregate_sync_accuracy(per_match)
    per_match["ALL"] = (agg_counts, agg_rates)
    return per_match


def collect_rows(group: str, value: float, result: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> list[dict]:
    rows: list[dict] = []
    for mid, (counts, rates) in result.items():
        for cat in rates.index:
            row = {"sweep_group": group, "target_value": value, "match_id": mid, "category": cat}
            row.update(rates.loc[cat].to_dict())
            row["mean_diff"] = counts.loc[cat, "mean_diff"]
            rows.append(row)
    return rows


def main() -> None:
    t0 = time.time()

    print("loading data ...")
    cache = load_data()

    print("=" * 72)
    print("baseline (all weights = 0.25)")
    print("=" * 72)
    baseline = run_one_config(cache, {})
    print(baseline["ALL"][1].round(4))

    all_rows: list[dict] = []

    for group in SWEEP_GROUPS:
        attr = GROUP_TO_ATTR[group]
        for value in SWEEP_VALUES:
            label = f"{group}={value:.2f}"
            print()
            print("=" * 72)
            if value == BASELINE_WEIGHT:
                print(f"[{label}] -- baseline reused")
                print("=" * 72)
                result = baseline
            else:
                print(f"[{label}]")
                print("=" * 72)
                result = run_one_config(cache, {attr: value})
            print(result["ALL"][1].round(4))
            all_rows.extend(collect_rows(group, value, result))

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(RESULT_DIR) / f"coeff_sweep_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print()
    print(f"saved: {out_path}")
    print(f"total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
