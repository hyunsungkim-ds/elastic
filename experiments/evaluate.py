import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sync import config
from sync.utils import collapse_events


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
    buffers: int | list[int] | np.ndarray = [0, 2, 5, 25, 50],
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
    buffers: int | list[int] | np.ndarray = [0, 2, 5, 25, 50],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute frame accuracy after collapsing control events.

    Assumes ``synced`` and ``annotated`` represent the same event sequence after
    collapsing controls, so row indices should match 1:1.
    """
    category_order = ["pass_like", "set_piece", "incoming", "minor"]

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
        receive_mask = annotated["event_cat"].isin(["pass_like", "set_piece"])
        synced.loc[receive_mask, "receive_status"] = _sync_status(
            synced.loc[receive_mask, "receive_frame_id"],
            annotated.loc[receive_mask, "receive_frame_id"],
        ).to_numpy()

    acc_counts = {}
    for cat in category_order:
        mask = annotated["event_cat"] == cat
        acc_counts[cat] = _frame_metrics(synced.loc[mask, "frame_id"], annotated.loc[mask, "frame_id"], buffers)

    acc_counts = pd.DataFrame.from_dict(acc_counts, orient="index")
    acc_counts.loc["event_start"] = _frame_metrics(synced["frame_id"], annotated["frame_id"], buffers)

    if include_receive:
        acc_counts.loc["event_end"] = _frame_metrics(
            synced.loc[receive_mask, "receive_frame_id"],
            annotated.loc[receive_mask, "receive_frame_id"],
            buffers,
        )

        acc_counts.loc["total"] = acc_counts.loc["event_start"] + acc_counts.loc["event_end"]
        total_denom = acc_counts.at["total", "total"]
        if total_denom > 0:
            start_sum = _sum_abs_diff(synced["frame_id"], annotated["frame_id"])
            end_sum = _sum_abs_diff(
                synced.loc[receive_mask, "receive_frame_id"],
                annotated.loc[receive_mask, "receive_frame_id"],
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


# ---------------------------------------------------------------------------
# Synchronization accuracy benchmark (paper Table 2)
#
# Runs the requested syncer on the input events produced by benchmark.py
# (data/sportec/event_corrected/{mid}.parquet), evaluates against the GT
# (data/sportec/event_synced/gt/{mid}.parquet), and optionally caches the
# synced output to data/sportec/event_synced/{method}/{mid}.parquet.
#
# --load reads a pre-saved synced parquet instead of running the syncer.
# databallpy is external-only and always loaded from disk.
#
# Run from repo root:
#     python experiments/evaluate.py --method elastic_nw [--save]
#     python experiments/evaluate.py --method elastic_nw --load
#     python experiments/evaluate.py --method databallpy

import argparse

MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY"]
INPUT_DIR = "data/sportec/event_corrected"
SYNCED_DIR = "data/sportec/event_synced"


def _run_syncer(method: str, input_events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    from sync import elastic_greedy, elastic_nw, etsy, schema

    if method == "elastic_nw":
        cols = list(schema.elastic_event_schema.columns.keys())
        return elastic_nw.ELASTIC_NW(input_events[cols], tracking, detect_controls=True).run()
    if method == "elastic_greedy":
        cols = list(schema.elastic_event_schema.columns.keys())
        return elastic_greedy.ELASTIC_Greedy(input_events[cols], tracking).run()
    if method == "etsy":
        cols = list(schema.etsy_event_schema.columns.keys())
        return etsy.ETSY(input_events[cols], tracking).run()
    raise ValueError(f"Unknown method: {method!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronization accuracy benchmark (paper Table 2).")
    parser.add_argument("--method", required=True, choices=["elastic_nw", "elastic_greedy", "etsy", "databallpy"])
    parser.add_argument("--load", action="store_true", help="Load pre-saved synced parquet.")
    parser.add_argument("--save", action="store_true", help="Save synced outputs as a parquet file.")
    args = parser.parse_args()

    load = args.load or args.method == "databallpy"
    if load and args.save:
        reason = "method=databallpy" if args.method == "databallpy" else "--load"
        print(f"Note: --save is ignored because {reason} loads from disk.")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    sportec_data_cls = None
    if not load:
        from tools.sportec_data import SportecData

        sportec_data_cls = SportecData
        if args.save:
            os.makedirs(f"{SYNCED_DIR}/{args.method}", exist_ok=True)

    per_match: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mid in MATCH_IDS:
        gt = pd.read_parquet(f"{SYNCED_DIR}/gt/{mid}.parquet")

        if load:
            synced = pd.read_parquet(f"{SYNCED_DIR}/{args.method}/{mid}.parquet")
        else:
            input_events = pd.read_parquet(f"{INPUT_DIR}/{mid}.parquet")
            tracking = sportec_data_cls(mid).format_tracking_for_syncer()
            synced = _run_syncer(args.method, input_events, tracking)
            if args.save:
                synced.to_parquet(f"{SYNCED_DIR}/{args.method}/{mid}.parquet")

        counts, rates, _ = compute_sync_accuracy(synced, gt)
        per_match[mid] = (counts, rates)
        print(f"\n=== {mid} ===")
        print(counts.round(3))

    total_counts, _ = aggregate_sync_accuracy(per_match)
    print(f"\n=== Total ({args.method}) ===")
    print(total_counts.round(3))


if __name__ == "__main__":
    main()
