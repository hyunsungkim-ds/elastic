"""Synchronization accuracy of each method on the Sportec benchmark (paper Table 2).

Also defines the benchmark constants, data loader, and accuracy metrics that the sweep
scripts import, so that every script reports the same numbers.

Input: syncer input events and GT events under UNSYNCED_DIR / GT_DIR (built by benchmark.py).
Output: per-match and total accuracy tables printed to stdout.

Run from the repo root:
    python experiments/evaluate.py --method elastic_nw
"""

import argparse
import os
import sys
import time

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sync import config, schema
from sync.utils import collapse_events

# Shared constants for the Sportec benchmark.
MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY"]
ANNOT_DIR = "data/sportec/event_corrected"  # per-annotator labels ({mid}_{annotator}.csv)
UNSYNCED_DIR = "benchmark/unsynced"  # events with unsynchronized timestamps ({mid}.parquet)
SYNCED_DIR = "benchmark/synced"  # cached synced outputs per method
GT_DIR = "benchmark/gt"  # ground-truth events ({mid}.parquet)
RESULT_DIR = "experiments/results"  # sweep result CSVs
FPS = 25  # tracking frame rate of the Sportec dataset

# Layout of the result tables: methods, event categories, and frame tolerances in report order.
SYNC_METHODS = ["etsy", "biermann", "databallpy", "elastic_greedy", "elastic_nw"]
EVENT_CATS = ["pass_like", "set_piece", "incoming", "minor"]
TIME_BUFFERS = [0, 2, 5, 25, 50]  # |pred - true| in frames; 0 is reported as "exact", t > 0 as "within_t"


def load_data(match_ids: list[str] = MATCH_IDS, margin: int = 0) -> dict[str, dict]:
    """Load input_events / gt (from cached parquet) + tracking (via SportecData) once per match.

    Input events come from {UNSYNCED_DIR}/{mid}.parquet and GT from {GT_DIR}/{mid}.parquet
    (both produced by experiments/benchmark.py), NOT rebuilt from _merged.csv.
    ``margin`` is forwarded to ``format_tracking_for_syncer``.
    """
    from tools.sportec_data import SportecData

    cache: dict[str, dict] = {}
    elastic_cols = list(schema.elastic_event_schema.columns.keys())
    for mid in match_ids:
        input_events = pd.read_parquet(f"{UNSYNCED_DIR}/{mid}.parquet")
        gt = pd.read_parquet(f"{GT_DIR}/{mid}.parquet")
        cache[mid] = {
            "input_events": input_events[elastic_cols],
            "gt": gt,
            "tracking": SportecData(mid).format_tracking_for_syncer(margin),
        }
    return cache


def _frame_metrics(pred_frames: pd.Series, true_frames: pd.Series, thresholds: list[int]) -> dict[str, float]:
    pred = pd.to_numeric(pd.Series(pred_frames).reset_index(drop=True), errors="coerce")
    true = pd.to_numeric(pd.Series(true_frames).reset_index(drop=True), errors="coerce")
    diff = (pred - true).abs()

    result: dict[str, float] = {"total": int(len(true)), "mean_diff": diff.mean()}
    for t in thresholds:
        col = "exact" if t == 0 else f"within_{t}"
        result[col] = int((diff <= t).sum())
    result["valid"] = int(pred.notna().sum())
    return result


def _sum_abs_diff(pred_frames: pd.Series, true_frames: pd.Series) -> float:
    pred = pd.to_numeric(pd.Series(pred_frames).reset_index(drop=True), errors="coerce")
    true = pd.to_numeric(pd.Series(true_frames).reset_index(drop=True), errors="coerce")
    return float((pred - true).abs().sum())


def _sync_status(
    pred_frames: pd.Series,
    true_frames: pd.Series,
    buffers: int | list[int] | np.ndarray = TIME_BUFFERS,
) -> pd.Series:
    pred = pd.to_numeric(pd.Series(pred_frames).reset_index(drop=True), errors="coerce")
    true = pd.to_numeric(pd.Series(true_frames).reset_index(drop=True), errors="coerce")
    diff = (pred - true).abs()

    buffers = sorted(set([0] + list(np.asarray(buffers).reshape(-1).tolist())), reverse=True)

    status = pd.Series("miss", index=pred.index, dtype="object")
    status[true.isna()] = pd.NA
    for t in buffers:
        label = "exact" if t == 0 else f"within_{t}"
        status[(true.notna()) & (diff <= t)] = label
    status[(true.notna()) & pred.isna()] = "miss"

    return status


def compute_sync_accuracy(
    synced: pd.DataFrame,
    annotated: pd.DataFrame,
    include_receive: bool = True,
    buffers: int | list[int] | np.ndarray = TIME_BUFFERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute frame accuracy after collapsing control events.

    Assumes ``synced`` and ``annotated`` represent the same event sequence after
    collapsing controls, so row indices should match 1:1.
    """
    synced = synced.copy()
    if "receive_frame_id" in synced.columns:
        synced = synced[~synced["spadl_type"].isin(config.EVENT_END)].reset_index(drop=True)
    elif synced["spadl_type"].isin(config.EVENT_END).any():
        synced = collapse_events(synced).reset_index(drop=True)
    else:
        # No event-end information available. Skip event_end evaluation automatically.
        include_receive = False

    annotated = annotated.copy()
    if "error_type" in annotated.columns:
        annotated = annotated[annotated["error_type"] != "false_positive"].reset_index(drop=True)
    annotated["event_cat"] = annotated["spadl_type"].map(config.EVENT_CAT_MAP)

    # Exclude rows whose spadl_type is not in any reported category (foul, take_on).
    # They are still synced upstream but excluded from evaluation per spec.
    keep_mask = annotated["event_cat"].notna()
    annotated = annotated.loc[keep_mask].reset_index(drop=True)
    synced = synced.loc[keep_mask.values].reset_index(drop=True)

    if "receive_frame_id" not in synced.columns:
        synced["receive_frame_id"] = np.nan
    if include_receive and "receive_frame_id" not in annotated.columns:
        raise ValueError("`annotated` must contain `receive_frame_id` when `include_receive=True`.")

    buffers = sorted(set([0] + list(np.asarray(buffers).reshape(-1).tolist())))

    synced = synced.copy()
    synced["event_status"] = _sync_status(synced["frame_id"], annotated["frame_id"])
    synced["receive_status"] = pd.NA
    if include_receive:
        kick_mask = annotated["event_cat"].isin(["pass_like", "set_piece"])
        synced.loc[kick_mask, "receive_status"] = _sync_status(
            synced.loc[kick_mask, "receive_frame_id"],
            annotated.loc[kick_mask, "receive_frame_id"],
        ).to_numpy()

    acc_counts = {}
    for cat in EVENT_CATS:
        mask = annotated["event_cat"] == cat
        acc_counts[cat] = _frame_metrics(synced.loc[mask, "frame_id"], annotated.loc[mask, "frame_id"], buffers)

    acc_counts = pd.DataFrame.from_dict(acc_counts, orient="index")
    acc_counts.loc["event_start"] = _frame_metrics(synced["frame_id"], annotated["frame_id"], buffers)

    if include_receive:
        acc_counts.loc["event_end"] = _frame_metrics(
            synced.loc[kick_mask, "receive_frame_id"],
            annotated.loc[kick_mask, "receive_frame_id"],
            buffers,
        )

        acc_counts.loc["total"] = acc_counts.loc["event_start"] + acc_counts.loc["event_end"]
        total_denom = acc_counts.at["total", "total"]
        if total_denom > 0:
            start_sum = _sum_abs_diff(synced["frame_id"], annotated["frame_id"])
            end_sum = _sum_abs_diff(
                synced.loc[kick_mask, "receive_frame_id"],
                annotated.loc[kick_mask, "receive_frame_id"],
            )
            acc_counts.at["total", "mean_diff"] = (start_sum + end_sum) / total_denom
        else:
            acc_counts.at["total", "mean_diff"] = np.nan
    else:
        acc_counts.loc["total"] = acc_counts.loc["event_start"]

    int_cols = ["total"] + ["exact" if t == 0 else f"within_{t}" for t in buffers] + ["valid"]
    acc_counts[int_cols] = acc_counts[int_cols].fillna(0).astype(int)
    acc_rates = acc_counts.drop(["total", "mean_diff"], axis=1).div(acc_counts["total"].replace(0, np.nan), axis=0)

    return acc_counts, acc_rates, synced


def aggregate_sync_accuracy(
    per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine per-match (acc_counts, acc_rates) into a single counts + rates table."""
    counts_list = [c for c, _ in per_match.values()]
    sample = counts_list[0]
    count_cols = [c for c in sample.columns if c != "mean_diff"]

    summed = sum(c[count_cols] for c in counts_list)
    mds = pd.DataFrame({mid: c["mean_diff"] * c["total"] for mid, (c, _) in per_match.items()}).sum(axis=1)
    summed["mean_diff"] = mds / summed["total"].replace(0, np.nan)

    rates = summed.drop(columns=["total", "mean_diff"]).div(summed["total"].replace(0, np.nan), axis=0)
    front = ["total", "mean_diff"]
    rest = [c for c in summed.columns if c not in front]
    summed = summed[front + rest]
    return summed, rates


def set_display_options() -> None:
    """Widen the pandas console output so an accuracy table prints on one line."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)


def accuracy_row(counts: pd.DataFrame, rates: pd.DataFrame, category: str) -> dict:
    """Flatten one category of an accuracy table into CSV columns."""
    row = rates.loc[category].to_dict()
    row["mean_diff"] = counts.loc[category, "mean_diff"]
    return row


def collect_rows(
    sweep_group: str,
    target_value: object,
    per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> list[dict]:
    """One CSV row per (match, category) for a single sweep config."""
    rows: list[dict] = []
    for mid, (counts, rates) in per_match.items():
        for cat in rates.index:
            row = {"sweep_group": sweep_group, "target_value": target_value, "match_id": mid, "category": cat}
            row.update(accuracy_row(counts, rates, cat))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Syncer runners


def _run_syncer(method: str, input_events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    from sync import elastic_greedy, elastic_nw, etsy

    if method == "elastic_nw":
        cols = list(schema.elastic_event_schema.columns.keys())
        return elastic_nw.ELASTIC_NW(input_events[cols], tracking).run()
    if method == "elastic_greedy":
        cols = list(schema.elastic_event_schema.columns.keys())
        return elastic_greedy.ELASTIC_Greedy(input_events[cols], tracking).run()
    if method == "etsy":
        cols = list(schema.etsy_event_schema.columns.keys())
        return etsy.ETSY(input_events[cols], tracking).run()
    raise ValueError(f"Unknown method: {method!r}")


def _run_biermann_lomocv() -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Leave-one-match-out CV: fit on the other matches' GT, then sync the held-out match.

    Tracking is loaded with a dead-ball margin of half the classification window so
    that windows at episode boundaries (e.g., set-piece kicks) are computable.
    Returns the synced events and the per-match sync runtime in seconds; the
    runtime covers inference only, excluding the training statistics.
    """
    from sync.biermann import HALF_WIN, BiermannModel, BiermannSync

    data = load_data(margin=HALF_WIN)
    t0 = time.perf_counter()
    stats = {mid: BiermannModel.match_statistics(d["input_events"], d["gt"], d["tracking"]) for mid, d in data.items()}
    print(f"Training statistics extracted in {time.perf_counter() - t0:.2f} s.")

    synced: dict[str, pd.DataFrame] = {}
    runtime: dict[str, float] = {}
    for mid, d in data.items():
        model = BiermannModel().fit([stats[m] for m in data if m != mid])
        t0 = time.perf_counter()
        synced[mid] = BiermannSync(d["input_events"], d["tracking"], model).run()
        runtime[mid] = time.perf_counter() - t0
    return synced, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronization accuracy benchmark (paper Table 2).")
    parser.add_argument("--method", required=True, choices=SYNC_METHODS)
    parser.add_argument("--load", action="store_true", help="Load pre-saved synced parquet.")
    parser.add_argument("--save", action="store_true", help="Save synced outputs as a parquet file.")
    args = parser.parse_args()

    # databallpy is external-only, so its synced output is always read from disk.
    load = args.load or args.method == "databallpy"
    if load and args.save:
        reason = "method=databallpy" if args.method == "databallpy" else "--load"
        print(f"Note: --save is ignored because {reason} loads from disk.")

    set_display_options()

    sportec_data_cls = None
    presynced: dict[str, pd.DataFrame] = {}
    runtime: dict[str, float] = {}
    if not load:
        if args.save:
            os.makedirs(f"{SYNCED_DIR}/{args.method}", exist_ok=True)
        if args.method == "biermann":
            presynced, runtime = _run_biermann_lomocv()
        else:
            from tools.sportec_data import SportecData

            sportec_data_cls = SportecData

    per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mid in MATCH_IDS:
        gt = pd.read_parquet(f"{GT_DIR}/{mid}.parquet")

        if load:
            synced = pd.read_parquet(f"{SYNCED_DIR}/{args.method}/{mid}.parquet")
        elif args.method == "biermann":
            synced = presynced[mid]
            if args.save:
                synced.to_parquet(f"{SYNCED_DIR}/{args.method}/{mid}.parquet")
        else:
            input_events = pd.read_parquet(f"{UNSYNCED_DIR}/{mid}.parquet")
            tracking = sportec_data_cls(mid).format_tracking_for_syncer()
            synced = _run_syncer(args.method, input_events, tracking)
            if args.save:
                synced.to_parquet(f"{SYNCED_DIR}/{args.method}/{mid}.parquet")

        counts, rates, _ = compute_sync_accuracy(synced, gt)
        per_match[mid] = (counts, rates)
        print(f"\n=== {mid} ===")
        print(counts.round(3))

    total_counts, total_rates = aggregate_sync_accuracy(per_match)
    print(f"\n=== Total ({args.method}) ===")
    print(total_counts.round(3))
    print()
    print(total_rates.round(3))

    if runtime:
        rt = pd.Series(runtime)
        print("\nPer-match sync runtime (s): " + ", ".join(f"{m}: {t:.2f}" for m, t in rt.items()))
        print(f"Mean ± std: {rt.mean():.2f} ± {rt.std():.2f} | total: {rt.sum():.2f}")


if __name__ == "__main__":
    main()
