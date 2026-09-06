"""Builds the ground-truth benchmark from the multi-annotator event labels (paper Table 1).

Input: per-annotator CSVs ({match_id}_merged.csv under INPUT_DIR) and Sportec tracking data.
Output: syncer input events and GT events as parquet under INPUT_DIR / GT_DIR, plus the
per-category inter-annotator agreement printed to stdout.

Run from the repo root:
    python experiments/benchmark.py
"""

from __future__ import annotations

import itertools
import os
import sys
from datetime import timedelta

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.evaluate import FPS, GT_DIR, INPUT_DIR, MATCH_IDS, set_display_options
from sync import config
from sync.utils import collapse_events

LOOKAHEAD = 30


def _discover_annotators(input_dir: str, match_id: str) -> list[str]:
    """Return the annotator tags present in ``{input_dir}/{match_id}_*.csv``.

    The annotation pipeline writes one ``{match_id}_{annotator}.csv`` per annotator
    plus the cross-rater products ``_aligned.csv`` and ``_merged.csv``. Stripping
    the leading ``{match_id}_`` from each filename and excluding the two products
    yields the annotator tags without hardcoding any names.
    """
    excluded = {"aligned", "merged"}
    prefix = f"{match_id}_"
    tags: list[str] = []
    for fname in sorted(os.listdir(input_dir)):
        if not (fname.startswith(prefix) and fname.endswith(".csv")):
            continue
        tag = fname[len(prefix) : -len(".csv")]
        if tag in excluded:
            continue
        tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# Timestamp helpers: the annotation tool writes "mm:ss.cc"


def _ts_to_sec(ts) -> float:
    if pd.isna(ts):
        return np.nan
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def _sec_to_ts(sec: float) -> str:
    if np.isnan(sec):
        return ""
    mm = int(sec // 60)
    ss = int(sec % 60)
    hh = int(round((sec - int(sec)) * 100))
    return f"{mm:02d}:{ss:02d}.{hh:02d}"


def _median_sec(row, annotators: list[str]) -> float:
    secs = [_ts_to_sec(row[f"{a}_ts"]) for a in annotators]
    return float(np.median(secs))


def _period_starts(tracking: pd.DataFrame, fps: int = FPS) -> tuple[dict, dict]:
    """Per-period "00:00.00" anchors derived from the tracking stream.

    Tracking may drop a few opening frames so the first row's timestamp can be
    > 0 (observed 0.04 in J03WN1 period 2); we subtract that offset to recover
    the canonical 0.00 anchor for both utc and frame_id.
    """
    first = (
        tracking.sort_values("frame_id")
        .groupby("period_id")
        .agg(
            track_utc=("utc_timestamp", "first"),
            track_frame=("frame_id", "first"),
            track_ts=("timestamp", "first"),
        )
    )
    starts_utc = {pid: row["track_utc"] - timedelta(seconds=float(row["track_ts"])) for pid, row in first.iterrows()}
    starts_frame = {pid: int(round(row["track_frame"] - float(row["track_ts"]) * fps)) for pid, row in first.iterrows()}
    return starts_utc, starts_frame


# ---------------------------------------------------------------------------
# Syncer input helpers: align the provider stream against the annotated events


def _match_score(
    provider_row: pd.Series,
    corrected_row: pd.Series,
    period_starts_utc: dict,
    annotators: list[str],
    threshold_sec: float = 30,
) -> bool:
    if provider_row["period_id"] != corrected_row["period_id"]:
        return False

    provider_sec = (provider_row["utc_timestamp"] - period_starts_utc[provider_row["period_id"]]).total_seconds()
    corrected_sec = _median_sec(corrected_row, annotators)
    if abs(provider_sec - corrected_sec) > threshold_sec:
        return False

    et = corrected_row["error_type"]
    same_player = provider_row["player_id"] == corrected_row["player_id"]
    same_type = provider_row["spadl_type"] == corrected_row["spadl_type"]
    if et == "spadl_type":
        return same_player
    if et == "player_id":
        return same_type
    else:
        return same_player and same_type


def _find_forward_provider(provider, ptr_p, target_corrected, period_starts_utc, annotators):
    for la in range(ptr_p + 1, min(ptr_p + 1 + LOOKAHEAD, len(provider))):
        if _match_score(provider.iloc[la], target_corrected, period_starts_utc, annotators):
            return la
    return None


def _find_forward_corrected(corrected, ptr_c, target_provider, period_starts_utc, annotators):
    for la in range(ptr_c + 1, min(ptr_c + 1 + LOOKAHEAD, len(corrected))):
        if corrected.iloc[la]["error_type"] == "missing":
            continue
        if _match_score(target_provider, corrected.iloc[la], period_starts_utc, annotators):
            return la
    return None


def _make_matched_row(provider_row, corrected_row) -> dict:
    return {
        "period_id": int(corrected_row["period_id"]),
        "utc_timestamp": provider_row["utc_timestamp"],
        "player_id": corrected_row["player_id"],
        "spadl_type": corrected_row["spadl_type"],
        "start_x": float(provider_row["start_x"]),
        "start_y": float(provider_row["start_y"]),
        "success": bool(corrected_row["outcome"]),
    }


def _make_missing_row(
    corrected_row,
    period_starts_utc: dict,
    period_starts_frame: dict,
    ball_track: pd.DataFrame,
    annotators: list[str],
    fps: int = FPS,
) -> dict:
    pid = int(corrected_row["period_id"])
    sec = _median_sec(corrected_row, annotators)
    utc = period_starts_utc[pid] + timedelta(seconds=sec)
    target_frame = int(round(period_starts_frame[pid] + sec * fps))
    if target_frame in ball_track.index:
        bx = ball_track.at[target_frame, "x"]
        by = ball_track.at[target_frame, "y"]
    else:
        nearest = ball_track.index[np.abs(ball_track.index.to_numpy() - target_frame).argmin()]
        bx = ball_track.at[nearest, "x"]
        by = ball_track.at[nearest, "y"]
    return {
        "period_id": pid,
        "utc_timestamp": utc,
        "player_id": corrected_row["player_id"],
        "spadl_type": corrected_row["spadl_type"],
        "start_x": float(bx),
        "start_y": float(by),
        "success": bool(corrected_row["outcome"]),
    }


# ---------------------------------------------------------------------------
# Benchmark construction and annotator agreement


def build_syncer_input_events(
    provider_events: pd.DataFrame,
    corrected_events: pd.DataFrame,
    tracking: pd.DataFrame,
    annotators: list[str],
    fps: int = FPS,
) -> pd.DataFrame:
    """Fuse provider events with the corrected (annotator-consensus) table into the syncer's input format.

    The corrected table holds the human-verified label set + ``error_type`` flags;
    the provider events carry ``utc_timestamp``, ``start_x``, ``start_y``.

    Returns a DataFrame with columns
    ``[period_id, utc_timestamp, player_id, spadl_type, start_x, start_y, success]``.
    """
    starts_utc, starts_frame = _period_starts(tracking, fps)
    ball_track = tracking[tracking["ball"]].drop_duplicates("frame_id").set_index("frame_id")

    provider = provider_events.reset_index(drop=True)
    end_mask = corrected_events["spadl_type"].isin(config.EVENT_END)
    fp_mask = corrected_events["error_type"] == "false_positive"
    corrected = corrected_events[~end_mask & ~fp_mask].reset_index(drop=True)

    ptr_p, ptr_c = 0, 0
    output: list[dict] = []
    while ptr_p < len(provider) or ptr_c < len(corrected):
        if ptr_c < len(corrected) and corrected.iloc[ptr_c]["error_type"] == "missing":
            output.append(
                _make_missing_row(
                    corrected.iloc[ptr_c],
                    starts_utc,
                    starts_frame,
                    ball_track,
                    annotators,
                    fps,
                )
            )
            ptr_c += 1
            continue
        if ptr_p < len(provider) and ptr_c < len(corrected):
            if _match_score(provider.iloc[ptr_p], corrected.iloc[ptr_c], starts_utc, annotators):
                output.append(_make_matched_row(provider.iloc[ptr_p], corrected.iloc[ptr_c]))
                ptr_p += 1
                ptr_c += 1
                continue
            fwd_p = _find_forward_provider(provider, ptr_p, corrected.iloc[ptr_c], starts_utc, annotators)
            fwd_c = _find_forward_corrected(corrected, ptr_c, provider.iloc[ptr_p], starts_utc, annotators)
            if fwd_p is not None and (fwd_c is None or (fwd_p - ptr_p) <= (fwd_c - ptr_c)):
                ptr_p = fwd_p
            elif fwd_c is not None:
                for j in range(ptr_c, fwd_c):
                    output.append(
                        _make_missing_row(
                            corrected.iloc[j],
                            starts_utc,
                            starts_frame,
                            ball_track,
                            annotators,
                            fps,
                        )
                    )
                ptr_c = fwd_c
            else:
                ptr_p += 1
        elif ptr_p < len(provider):
            ptr_p = len(provider)
        else:
            for j in range(ptr_c, len(corrected)):
                output.append(
                    _make_missing_row(
                        corrected.iloc[j],
                        starts_utc,
                        starts_frame,
                        ball_track,
                        annotators,
                        fps,
                    )
                )
            ptr_c = len(corrected)

    cols = ["period_id", "utc_timestamp", "player_id", "spadl_type", "start_x", "start_y", "success"]
    return pd.DataFrame(output, columns=cols)


def build_gt_events(
    corrected_events: pd.DataFrame,
    tracking: pd.DataFrame,
    annotators: list[str],
    fps: int = FPS,
) -> pd.DataFrame:
    """Build the ground-truth event table consumed by ``compute_sync_accuracy``.

    Steps:
      1. Drop ``false_positive`` rows.
      2. Compute median frame_id from the per-annotator timestamps.
      3. Run ``collapse_events`` so ``control``/``out``/``goal`` rows fold into
         the preceding pass-like event's ``receive_*`` fields.
    """
    _, starts_frame = _period_starts(tracking, fps)

    df = corrected_events[corrected_events["error_type"] != "false_positive"].copy().reset_index(drop=True)
    df["_sec"] = df.apply(lambda r: _median_sec(r, annotators), axis=1)
    df["synced_ts"] = df["_sec"].map(_sec_to_ts)
    df["frame_id"] = df.apply(lambda r: int(round(starts_frame[int(r["period_id"])] + r["_sec"] * fps)), axis=1)
    df["success"] = df["outcome"].astype(bool)
    df["offside"] = False

    keep_cols = [
        "period_id",
        "player_id",
        "spadl_type",
        "frame_id",
        "synced_ts",
        "success",
        "offside",
    ]
    return collapse_events(df[keep_cols]).reset_index(drop=True)


def compute_annot_reliability(
    corrected_events: pd.DataFrame,
    tracking: pd.DataFrame,
    annotators: list[str],
    fps: int = FPS,
) -> pd.DataFrame:
    """Per-category inter-annotator agreement.

    Runs ``collapse_events`` once per annotator so the event-unit definitions
    (pass_like / set_piece / incoming / minor as event_starts, pass_like +
    set_piece receive frames as event_ends) match exactly what
    ``compute_sync_accuracy`` compares. Category counts here therefore align
    one-for-one with the counts emitted by ``compute_sync_accuracy(synced, gt)``.

    Returns a DataFrame indexed by event_cat (pass_like, set_piece, incoming,
    minor, event_end, total) with columns: total, mean_diff, exact{3,2}_count,
    close{3,2}_count, exact{3,2}_rate, close{3,2}_rate.
    """
    _, starts_frame = _period_starts(tracking, fps)

    df = corrected_events[corrected_events["error_type"] != "false_positive"].copy().reset_index(drop=True)
    df["success"] = df["outcome"].astype(bool)
    df["offside"] = False

    for ann in annotators:
        df[f"{ann}_frame"] = df.apply(
            lambda r: starts_frame[int(r["period_id"])] + _ts_to_sec(r[f"{ann}_ts"]) * fps,
            axis=1,
        ).round()

    # Run collapse_events once per annotator. spadl_type/player_id/period_id
    # are shared, so the resulting row structure is identical across the three;
    # only frame_id / receive_frame_id values differ.
    cols_for_collapse = [
        "period_id",
        "player_id",
        "spadl_type",
        "frame_id",
        "synced_ts",
        "success",
        "offside",
    ]
    collapsed_per_ann = {}
    for ann in annotators:
        df_ann = df.copy()
        df_ann["frame_id"] = df_ann[f"{ann}_frame"]
        df_ann["synced_ts"] = df_ann[f"{ann}_ts"]
        collapsed_per_ann[ann] = collapse_events(df_ann[cols_for_collapse]).reset_index(drop=True)

    collapsed = collapsed_per_ann[annotators[0]][["period_id", "spadl_type"]].copy()
    collapsed["event_cat"] = collapsed["spadl_type"].map(config.EVENT_CAT_MAP)
    for ann in annotators:
        collapsed[f"{ann}_frame"] = collapsed_per_ann[ann]["frame_id"].values
        collapsed[f"{ann}_receive_frame"] = collapsed_per_ann[ann]["receive_frame_id"].values

    start_cols = [f"{a}_frame" for a in annotators]
    event_starts = collapsed[start_cols + ["event_cat"]].copy()
    event_starts.columns = annotators + ["event_cat"]

    end_cols = [f"{a}_receive_frame" for a in annotators]
    end_mask = collapsed["event_cat"].isin(["pass_like", "set_piece"])
    event_ends = collapsed.loc[end_mask, end_cols].copy()
    event_ends.columns = annotators
    event_ends["event_cat"] = "event_end"

    stacked = pd.concat([event_starts, event_ends], ignore_index=True)
    stacked = stacked.dropna(subset=["event_cat"]).reset_index(drop=True)

    stacked["n_notna"] = stacked[annotators].notna().sum(axis=1)
    # NaN-only rows: median falls back to 0 (placeholder, won't match any frame).
    stacked["median"] = stacked[annotators].median(axis=1, skipna=True).fillna(0)

    stacked["exact3"] = (stacked["n_notna"] == 3) & (stacked[annotators].nunique(axis=1) == 1)
    stacked["exact2"] = (stacked["n_notna"] >= 2) & (stacked[annotators].nunique(axis=1) <= 2)
    within2 = (stacked[annotators].sub(stacked["median"], axis=0).abs() <= 2).sum(axis=1)
    stacked["close3"] = within2 == 3
    stacked["close2"] = within2 >= 2

    pair_diffs = [(stacked[a] - stacked[b]).abs().to_numpy() for a, b in itertools.combinations(annotators, 2)]
    stacked["frame_diff"] = np.stack(pair_diffs).mean(axis=0)

    groups = ["pass_like", "set_piece", "incoming", "minor", "event_end"]
    out = {}
    for cat in groups:
        sub = stacked[stacked["event_cat"] == cat]
        out[cat] = _reliability_row(sub)
    out["total"] = _reliability_row(stacked)

    rel = pd.DataFrame.from_dict(out, orient="index")
    for c in ["exact3", "exact2", "close3", "close2"]:
        rel[f"{c}_rate"] = rel[f"{c}_count"] / rel["total"].replace(0, np.nan)
    return rel


def _reliability_row(sub: pd.DataFrame) -> dict:
    return {
        "total": len(sub),
        "mean_diff": sub["frame_diff"].mean() if len(sub) else np.nan,
        "exact3_count": int(sub["exact3"].sum()),
        "exact2_count": int(sub["exact2"].sum()),
        "close3_count": int(sub["close3"].sum()),
        "close2_count": int(sub["close2"].sum()),
    }


def aggregate_annot_reliability(per_match: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-match reliability tables (counts) and recompute rates."""
    count_cols = ["total", "exact3_count", "exact2_count", "close3_count", "close2_count"]
    summed = sum(df[count_cols] for df in per_match.values())

    diffs = pd.DataFrame({mid: df["mean_diff"] * df["total"] for mid, df in per_match.items()}).sum(axis=1)
    summed["mean_diff"] = diffs / summed["total"].replace(0, np.nan)

    for c in ["exact3", "exact2", "close3", "close2"]:
        summed[f"{c}_rate"] = summed[f"{c}_count"] / summed["total"].replace(0, np.nan)
    return summed


def main() -> None:
    from tools.sportec_data import SportecData

    set_display_options()

    annotators = _discover_annotators(INPUT_DIR, MATCH_IDS[0])
    os.makedirs(GT_DIR, exist_ok=True)

    per_match: dict[str, pd.DataFrame] = {}
    for mid in MATCH_IDS:
        sportec = SportecData(mid)
        tracking = sportec.format_tracking_for_syncer()
        provider = sportec.format_events_for_syncer().reset_index(drop=True)
        corrected = pd.read_csv(f"{INPUT_DIR}/{mid}_merged.csv")

        build_syncer_input_events(provider, corrected, tracking, annotators).to_parquet(f"{INPUT_DIR}/{mid}.parquet")
        build_gt_events(corrected, tracking, annotators).to_parquet(f"{GT_DIR}/{mid}.parquet")

        per_match[mid] = compute_annot_reliability(corrected, tracking, annotators)
        print(f"\n=== {mid} ===")
        print(per_match[mid].round(3))

    total = aggregate_annot_reliability(per_match)
    print("\n=== Total ===")
    print(total.round(3))


if __name__ == "__main__":
    main()
