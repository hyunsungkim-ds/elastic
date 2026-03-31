import numpy as np
import pandas as pd

from sync import config, utils


def _to_unix_seconds(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce")
    out = np.full(len(dt), np.nan, dtype=np.float64)
    mask = dt.notna().to_numpy()
    if mask.any():
        out[mask] = dt[mask].astype("int64").to_numpy(dtype=np.float64) / 1e9
    return out


def _build_pair_scores(a: pd.DataFrame, b: pd.DataFrame, max_time_diff_sec: float) -> tuple[np.ndarray, np.ndarray]:
    a_type = a["spadl_type"].to_numpy(dtype=object)
    b_type = b["spadl_type"].to_numpy(dtype=object)
    a_player = a["player_id"].to_numpy(dtype=object)
    b_player = b["player_id"].to_numpy(dtype=object)

    type_score = (a_type[:, None] == b_type[None, :]).astype(np.float64) * 25.0
    player_score = (a_player[:, None] == b_player[None, :]).astype(np.float64) * 25.0

    a_ts = _to_unix_seconds(a["utc_timestamp"])
    b_ts = _to_unix_seconds(b["utc_timestamp"])
    dt = np.abs(a_ts[:, None] - b_ts[None, :])

    # 같은 시각이면 50, 2초 이상 차이면 0 (선형 감소)
    ts_score = np.where(
        np.isfinite(dt),
        np.clip(1.0 - (dt / max_time_diff_sec), 0.0, 1.0) * 50.0,
        0.0,
    )

    total_score = type_score + player_score + ts_score
    return total_score, dt


def _needleman_wunsch(a: pd.DataFrame, b: pd.DataFrame, gap_penalty: float, max_time_diff_sec: float) -> pd.DataFrame:
    n, m = len(a), len(b)

    cols = [
        "a_index",
        "b_index",
        "op",
        "pair_score",
        "time_diff_sec",
        "type_match",
        "player_match",
        "time_score",
    ]
    if n == 0 and m == 0:
        return pd.DataFrame(columns=cols)

    score_matrix, dt_matrix = _build_pair_scores(a, b, max_time_diff_sec)

    # DP / traceback
    dp = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    tb = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 0=diag, 1=up, 2=left
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty
        tb[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty
        tb[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1, j - 1] + score_matrix[i - 1, j - 1]
            up = dp[i - 1, j] + gap_penalty
            left = dp[i, j - 1] + gap_penalty

            best = diag
            move = 0
            if up > best:
                best = up
                move = 1
            if left > best:
                best = left
                move = 2

            dp[i, j] = best
            tb[i, j] = move

    a_idx = a.index.to_numpy()
    b_idx = b.index.to_numpy()
    a_type = a["spadl_type"].to_numpy(dtype=object)
    b_type = b["spadl_type"].to_numpy(dtype=object)
    a_player = a["player_id"].to_numpy(dtype=object)
    b_player = b["player_id"].to_numpy(dtype=object)

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        move = tb[i, j]

        if move == 0 and i > 0 and j > 0:
            ai, bj = i - 1, j - 1
            dt = dt_matrix[ai, bj]
            time_score = max(0.0, 1.0 - (dt / max_time_diff_sec)) * 50.0 if np.isfinite(dt) else 0.0
            aligned.append(
                {
                    "a_index": a_idx[ai],
                    "b_index": b_idx[bj],
                    "op": "match",
                    "pair_score": float(score_matrix[ai, bj]),
                    "time_diff_sec": float(dt) if np.isfinite(dt) else np.nan,
                    "type_match": bool(a_type[ai] == b_type[bj]),
                    "player_match": bool(a_player[ai] == b_player[bj]),
                    "time_score": float(time_score),
                }
            )
            i -= 1
            j -= 1

        elif move == 1 and i > 0:
            ai = i - 1
            aligned.append(
                {
                    "a_index": a_idx[ai],
                    "b_index": pd.NA,
                    "op": "false_positive",
                    "pair_score": np.nan,
                    "time_diff_sec": np.nan,
                    "type_match": pd.NA,
                    "player_match": pd.NA,
                    "time_score": np.nan,
                }
            )
            i -= 1

        else:
            bj = j - 1
            aligned.append(
                {
                    "a_index": pd.NA,
                    "b_index": b_idx[bj],
                    "op": "missing",
                    "pair_score": np.nan,
                    "time_diff_sec": np.nan,
                    "type_match": pd.NA,
                    "player_match": pd.NA,
                    "time_score": np.nan,
                }
            )
            j -= 1

    aligned.reverse()
    return pd.DataFrame(aligned, columns=cols)


def align_with_corrected_events(
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    corrected: pd.DataFrame,
    gap_penalty: float = -30.0,
    max_time_diff_sec: float = 2.0,
) -> pd.DataFrame:
    periods = sorted(set(events["period_id"].dropna().unique()) | set(corrected["period_id"].dropna().unique()))

    period_alignment = []
    for p in periods:
        a = events[events["period_id"] == p]
        b = corrected[corrected["period_id"] == p]
        part = _needleman_wunsch(a, b, gap_penalty, max_time_diff_sec)
        part.insert(0, "period_id", p)
        period_alignment.append(part)

    alignment_df = pd.concat(period_alignment, ignore_index=True)
    cols = [
        "player_id",
        "spadl_type",
        "frame_id",
        "synced_ts",
        "receiver_id",
        "receive_frame_id",
        "receive_ts",
        "success",
        "offside",
        "error_type",
    ]
    alignment_df = alignment_df.join(corrected[cols], on="b_index")

    nw_cols = ["a_index", "b_index", "op", "pair_score", "time_diff_sec", "type_match", "player_match", "time_score"]
    alignment_df = pd.merge(
        alignment_df[alignment_df["error_type"] != "false_positive"],
        events[["utc_timestamp", "start_x", "start_y"]],
        left_on="a_index",
        right_index=True,
        how="left",
    ).drop(nw_cols, axis=1)

    start_utc = tracking.at[0, "utc_timestamp"]
    missing = alignment_df[alignment_df["error_type"] == "missing"].copy()
    missing["utc_timestamp"] = missing["frame_id"].apply(utils.frame_to_utc_timestamp, args=(start_utc,))
    alignment_df.loc[missing.index, "utc_timestamp"] = missing["utc_timestamp"]

    ball_tracking = tracking[tracking["ball"]].copy().set_index("frame_id")
    alignment_df.loc[missing.index, "start_x"] = ball_tracking.loc[missing["frame_id"], "x"]
    alignment_df.loc[missing.index, "start_y"] = ball_tracking.loc[missing["frame_id"], "y"]

    alignment_df = alignment_df[alignment_df["error_type"] != "false_positive"].copy()
    return alignment_df.reset_index(drop=True)


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


def collapse_events(events: pd.DataFrame, tracking: pd.DataFrame | None = None) -> pd.DataFrame:
    """Collapse NW event sequences into Greedy-style receive annotations.

    Explicit ``control``/``out`` rows are folded into the previous pass-like event's
    ``receive_*`` fields. In addition, direct receive markers that do not have
    a separate control row are also folded back:
    - incoming events (same semantics as ``ReceiveDetector._detect_receive``)
    - one-touch pass-like events whose preceding control was removed because it
      aligned to the same frame as the action itself
    - bad_touch immediately following a pass-like event
    """
    events = events.copy().reset_index(drop=True)
    if "spadl_type" not in events.columns:
        raise ValueError("`events` must contain `spadl_type`.")

    time_col = "timestamp" if "timestamp" in events.columns else "synced_ts" if "synced_ts" in events.columns else None
    pass_like_types = set(config.PASS_LIKE_OPEN + config.SET_PIECE)
    direct_receive_types = set(config.INCOMING + ["shot_block", "keeper_punch", "bad_touch", "tackle"])
    one_touch_types = {"pass", "cross", "shot", "clearance", "take_on", "dispossessed"}
    collapsed_rows = []

    def _same_episode(prev_event: dict, current_row: pd.Series) -> bool:
        if "episode_id" in current_row.index and "episode_id" in prev_event:
            return (
                pd.notna(prev_event["episode_id"])
                and pd.notna(current_row["episode_id"])
                and (prev_event["episode_id"] == current_row["episode_id"])
            )
        return prev_event.get("period_id") == current_row.get("period_id")

    def _can_assign_receive(prev_event: dict | None, current_row: pd.Series) -> bool:
        if prev_event is None:
            return False
        if prev_event.get("spadl_type") not in pass_like_types:
            return False
        if pd.isna(prev_event.get("frame_id")):
            return False
        if pd.notna(prev_event.get("receive_frame_id")):
            return False
        return _same_episode(prev_event, current_row)

    def _assign_receive(prev_event: dict, current_row: pd.Series) -> None:
        prev_event["receiver_id"] = current_row["player_id"]
        prev_event["receive_frame_id"] = current_row["frame_id"]
        prev_event["receive_ts"] = current_row[time_col] if time_col is not None else np.nan

    for _, row in events.iterrows():
        if row["spadl_type"] in ["control", "out"]:
            if _can_assign_receive(collapsed_rows[-1] if collapsed_rows else None, row):
                _assign_receive(collapsed_rows[-1], row)
            continue

        if _can_assign_receive(collapsed_rows[-1] if collapsed_rows else None, row):
            if row["spadl_type"] in direct_receive_types:
                _assign_receive(collapsed_rows[-1], row)
            elif row["spadl_type"] in one_touch_types and collapsed_rows[-1]["player_id"] != row["player_id"]:
                # NW can drop a control row when reception and next action align
                # to the same frame. In that case, the action frame is the
                # previous pass-like event's receive frame.
                _assign_receive(collapsed_rows[-1], row)

        collapsed_rows.append(
            {
                "period_id": row["period_id"],
                "episode_id": row["episode_id"] if "episode_id" in row.index else np.nan,
                "player_id": row["player_id"],
                "spadl_type": row["spadl_type"],
                "frame_id": row["frame_id"],
                "synced_ts": row[time_col] if time_col is not None else np.nan,
                "receiver_id": np.nan,
                "receive_frame_id": np.nan,
                "receive_ts": np.nan,
                "success": row["success"],
                "offside": False,
            }
        )

    collapsed = pd.DataFrame(collapsed_rows)

    if tracking is not None and not collapsed.empty:
        ball_tracking = tracking[tracking["ball"]].drop_duplicates("frame_id").set_index("frame_id")
        collapsed["start_x"] = collapsed["frame_id"].map(ball_tracking["x"])
        collapsed["start_y"] = collapsed["frame_id"].map(ball_tracking["y"])

    return collapsed.reset_index(drop=True)


def _nearest_cand_diff(frame_ids: pd.Series, cand_frame_ids: np.ndarray) -> np.ndarray:
    out = np.full(len(frame_ids), np.nan)
    if len(cand_frame_ids) == 0:
        return out
    valid_mask = frame_ids.notna().to_numpy()
    if valid_mask.any():
        vals = frame_ids[valid_mask].to_numpy(dtype=np.float64)
        idx = np.searchsorted(cand_frame_ids, vals)
        diff_right = np.abs(cand_frame_ids[np.clip(idx, 0, len(cand_frame_ids) - 1)] - vals)
        diff_left = np.abs(cand_frame_ids[np.clip(idx - 1, 0, len(cand_frame_ids) - 1)] - vals)
        out[valid_mask] = np.minimum(diff_left, diff_right)
    return out


def _coverage_at_thresholds(diffs: np.ndarray, thresholds: list[int]) -> pd.Series:
    valid = diffs[~np.isnan(diffs)]
    if len(valid) == 0:
        return pd.Series({t: np.nan for t in thresholds}, dtype=float)
    return pd.Series({t: float((valid <= t).mean()) for t in thresholds}, dtype=float)


def calculate_candidate_coverage(
    cand_frames: pd.DataFrame,
    true_events: pd.DataFrame,
    buffers: int | list[int] | np.ndarray = [0, 2, 5, 10],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether candidate frames cover true event frames.

    For each true event, computes the frame difference to the nearest candidate
    frame. Returns per-event differences and coverage rates broken down by event
    category and buffer threshold.

    Parameters
    ----------
    cand_frames:
        Output of ``ELASTIC_NW.find_candidate_frames``; must contain a
        ``frame_id`` column.
    true_events:
        Ground-truth events; must contain ``frame_id`` and ``spadl_type``
        columns. ``receive_frame_id`` is used for receive coverage when present.
    buffers:
        Tolerance in frames, given as a single integer or a list/array of
        integers. Events whose nearest candidate frame is within a threshold
        are considered "covered" at that threshold. Buffer 0 (exact match) is
        always included in the result regardless of the input.

    Returns
    -------
    coverage : pd.DataFrame
        Coverage rate (0–1) for each event category (rows) at each buffer threshold (columns).
    diffs : pd.DataFrame
        Per-event distance to the nearest candidate frame, indexed by
        ``true_events.index``, with columns ``frame_id`` and ``receive_frame_id``.
    """
    cand_frame_ids = np.sort(cand_frames["frame_id"].dropna().unique().astype(np.float64))
    thresholds = sorted(set([0] + list(np.asarray(buffers).reshape(-1).tolist())))

    frame_diffs = _nearest_cand_diff(pd.to_numeric(true_events["frame_id"], errors="coerce"), cand_frame_ids)
    receive_diffs = (
        _nearest_cand_diff(pd.to_numeric(true_events["receive_frame_id"], errors="coerce"), cand_frame_ids)
        if "receive_frame_id" in true_events.columns
        else np.full(len(true_events), np.nan)
    )
    diffs = pd.DataFrame({"frame_diff": frame_diffs, "receive_frame_diff": receive_diffs}, index=true_events.index)

    category_map = (
        {x: "pass_like" for x in config.PASS_LIKE_OPEN}
        | {x: "set_piece" for x in config.SET_PIECE}
        | {x: "incoming" for x in config.INCOMING}
        | {x: "minor" for x in config.MINOR}
    )
    event_cat = true_events["spadl_type"].map(category_map)

    coverage_rows = {}
    for cat in ["pass_like", "set_piece", "incoming", "minor"]:
        mask = (event_cat == cat).to_numpy()
        coverage_rows[cat] = _coverage_at_thresholds(frame_diffs[mask], thresholds)

    receive_mask = event_cat.isin({"pass_like", "set_piece"}).to_numpy()
    coverage_rows["receive"] = _coverage_at_thresholds(receive_diffs[receive_mask], thresholds)
    coverage_rows["overall"] = _coverage_at_thresholds(frame_diffs, thresholds)

    col_names = {t: "exact" if t == 0 else f"within_{t}" for t in thresholds}
    coverage = pd.DataFrame(coverage_rows).T.rename(columns=col_names)

    return coverage, diffs


def calculate_accuracy(
    synced: pd.DataFrame,
    annotated: pd.DataFrame,
    include_receive: bool = True,
    buffers: int | list[int] | np.ndarray = [0, 2, 5, 25, 50],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute frame accuracy after collapsing control events.

    Assumes ``synced`` and ``corrected`` represent the same event sequence after
    collapsing controls, so row indices should match 1:1.
    """
    category_map = (
        {x: "pass_like" for x in config.PASS_LIKE_OPEN}
        | {x: "set_piece" for x in config.SET_PIECE}
        | {x: "incoming" for x in config.INCOMING}
        | {x: "minor" for x in config.MINOR}
    )
    category_order = ["pass_like", "set_piece", "incoming", "minor"]

    synced = synced.copy()
    # synced = synced[~synced["spadl_type"].isin(["take_on", "second_take_on"])].copy().reset_index(drop=True)
    if "receive_frame_id" not in synced.columns:
        synced = collapse_events(synced).reset_index(drop=True)

    annotated = annotated.copy()
    # annotated = annotated[~annotated["spadl_type"].isin(["take_on", "second_take_on"])].reset_index(drop=True)
    if "error_type" in annotated.columns:
        annotated = annotated[annotated["error_type"] != "false_positive"].reset_index(drop=True)
    annotated["event_cat"] = annotated["spadl_type"].map(category_map)

    if "receive_frame_id" not in synced.columns:
        synced["receive_frame_id"] = np.nan
    if include_receive and "receive_frame_id" not in annotated.columns:
        raise ValueError("`corrected` must contain `receive_frame_id` when `include_receive=True`.")

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
