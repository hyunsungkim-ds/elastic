from datetime import datetime, timedelta
from typing import Callable, Union

import numpy as np
import pandas as pd

from sync.config import EVENT_END, INCOMING, PASS_LIKE_OPEN, PITCH_X, PITCH_Y, SET_PIECE


def frame_to_utc_timestamp(frame: float, start_utc: datetime, fps=25) -> datetime:
    return start_utc + timedelta(seconds=frame / fps) if not np.isnan(frame) else np.nan


def seconds_to_timestamp(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{int(seconds):02d}{f'{seconds % 1:.2f}'[1:]}"


def timestamp_to_seconds(t: str) -> float:
    return float(t[:2]) * 60 + float(t[3:]) if isinstance(t, str) else np.nan


def linear_scoring_func(min_input: float, max_input: float, increasing=False) -> Callable:
    assert min_input < max_input

    def func(x: float) -> float:
        if increasing:
            return (x - min_input) / (max_input - min_input)
        else:
            return 1 - (x - min_input) / (max_input - min_input)

    return lambda x: np.maximum(0, np.minimum(1, func(x)))


# Scoring functions for ELASTIC
player_dist_func = linear_scoring_func(0, 3, increasing=False)
out_dist_func = linear_scoring_func(0, 1, increasing=False)
player_speed_func = linear_scoring_func(0, 5, increasing=True)
player_accel_func = linear_scoring_func(0, 5, increasing=True)
ball_accel_func = linear_scoring_func(0, 30, increasing=True)
kick_dist_func = linear_scoring_func(0, 3, increasing=True)
angle_change_func = linear_scoring_func(0, 1, increasing=False)  # increasing from 0 to pi in radian
frame_delay_func = linear_scoring_func(0, 125, increasing=False)
positive_slope_penalty = linear_scoring_func(0, 0.28, increasing=False)  # 7 m/s at 25 fps
negative_slope_penalty = linear_scoring_func(-0.28, 0, increasing=True)  # 7 m/s at 25 fps


# Per-term weights for nw_score_major / nw_score_minor.
# Module-level so experiments can override them at runtime; default 0.25 preserves original behavior.
BA_WEIGHT = 0.25  # ball_accel (major + minor)
PBD_WEIGHT = 0.25  # player_dist (major + minor)
KD_WEIGHT = 0.25  # kick_dist (major + minor)
PBDS_WEIGHT = 0.25  # player_dist_slope (major only)
OD_WEIGHT = 0.25  # oppo_dist (minor only)


def nw_score_major(features: pd.Series | pd.DataFrame, player_id: str, incoming: bool = False) -> float | np.ndarray:
    dist_func = out_dist_func if player_id.startswith(("out_", "goal_")) else player_dist_func
    slope_func = negative_slope_penalty if incoming else positive_slope_penalty
    slope_col = "post_slope" if incoming else "pre_slope"
    kick_dist_col = "pre_kick_dist" if incoming else "post_kick_dist"

    if isinstance(features, pd.DataFrame):
        scores = np.zeros(len(features), dtype=float)
        if scores.size == 0:
            return scores

        mask = features["player_id"] == player_id
        if not mask.any():
            return scores

        features = features.loc[mask]
        player_dist_score = PBD_WEIGHT * dist_func(features["player_dist"].to_numpy())
        player_dist_slope_score = PBDS_WEIGHT * slope_func(features[slope_col].to_numpy())
        kick_dist_score = KD_WEIGHT * kick_dist_func(features[kick_dist_col].to_numpy())
        ball_accel_score = BA_WEIGHT * ball_accel_func(features["ball_accel"].to_numpy())
        scores[mask.to_numpy()] = player_dist_score + player_dist_slope_score + kick_dist_score + ball_accel_score
        return scores

    elif features["player_id"] == player_id:  # isinstance(features, pd.Series)
        player_dist_score = PBD_WEIGHT * dist_func(features["player_dist"])
        player_dist_slope_score = PBDS_WEIGHT * slope_func(features[slope_col])
        kick_dist_score = KD_WEIGHT * kick_dist_func(features[kick_dist_col])
        ball_accel_score = BA_WEIGHT * ball_accel_func(features["ball_accel"])
        return player_dist_score + player_dist_slope_score + kick_dist_score + ball_accel_score

    else:
        return 0.0


def nw_score_minor(features: pd.Series | pd.DataFrame, player_id: str, incoming: bool = False) -> float | np.ndarray:
    kick_dist_col = "pre_kick_dist" if incoming else "post_kick_dist"

    if isinstance(features, pd.DataFrame):
        scores = np.zeros(len(features), dtype=float)
        if scores.size == 0:
            return scores

        mask = features["player_id"] == player_id
        if not mask.any():
            return scores

        features = features.loc[mask]
        ball_accel_score = BA_WEIGHT * ball_accel_func(features["ball_accel"].to_numpy())
        player_dist_score = PBD_WEIGHT * player_dist_func(features["player_dist"].to_numpy())
        oppo_dist_score = OD_WEIGHT * player_dist_func(features["oppo_dist"].to_numpy())
        kick_dist_score = KD_WEIGHT * kick_dist_func(features[kick_dist_col].to_numpy())
        scores[mask.to_numpy()] = ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score
        return scores

    elif features["player_id"] == player_id:  # isinstance(features, pd.Series)
        ball_accel_score = BA_WEIGHT * ball_accel_func(features["ball_accel"])
        player_dist_score = PBD_WEIGHT * player_dist_func(features["player_dist"])
        oppo_dist_score = OD_WEIGHT * player_dist_func(features["oppo_dist"])
        kick_dist_score = KD_WEIGHT * kick_dist_func(features[kick_dist_col])
        return ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score

    else:
        return 0.0


def nw_score_takeon(features: pd.DataFrame, player_id: str, incoming: bool = False) -> np.ndarray:
    scores = np.zeros(len(features), dtype=float)
    mask = features["player_id"] == player_id
    if not mask.any():
        return scores

    f = features.loc[mask]
    scores[mask.to_numpy()] = (
        0.20 * ball_accel_func(f["ball_accel"].to_numpy() * 2)
        + 0.20 * player_speed_func(f["max_speed"].fillna(0).to_numpy())
        # + 0.20 * player_speed_func(f["delta_speed"].fillna(0).to_numpy() * 2)
        + 0.20 * player_dist_func(f["oppo_dist"].fillna(10).to_numpy() - 3)
        + 0.40 * angle_change_func(f["angle_change"].fillna(1).to_numpy())
    )
    return scores


def greedy_score_major(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 0.25 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 0.25 * player_dist_func(features["player_dist"].values)
    kick_dist_score = 0.25 * kick_dist_func(features["kick_dist"].values)
    frame_delay_score = 0.25 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + kick_dist_score + frame_delay_score


def greedy_score_tackle(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 0.20 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 0.20 * player_dist_func(features["player_dist"].values)
    oppo_dist_score = 0.20 * player_dist_func(features["oppo_dist"].values)
    kick_dist_score = 0.20 * kick_dist_func(features["kick_dist"].values)
    frame_delay_score = 0.20 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score + frame_delay_score


def greedy_score_takeon(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 0.20 * ball_accel_func(features["ball_accel"].values * 2)
    max_speed_score = 0.20 * player_speed_func(features["max_speed"].values)
    delta_speed_score = 0.20 * player_speed_func(features["delta_speed"].values * 2)
    oppo_dist_score = 0.20 * player_dist_func(features["oppo_dist"].values - 3)
    angle_change_score = 0.20 * angle_change_func(features["angle_change"].values)
    return ball_accel_score + max_speed_score + delta_speed_score + oppo_dist_score + angle_change_score


def greedy_score_dispossessed(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 1 / 3 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 1 / 3 * player_dist_func(features["player_dist"].values)
    kick_dist_score = 1 / 3 * kick_dist_func(features["kick_dist"].values)
    return ball_accel_score + player_dist_score + kick_dist_score


def greedy_score_receive(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 0.25 * ball_accel_func(features["ball_accel"].values)
    closest_dist_score = 0.25 * player_dist_func(features["closest_dist"].values)
    next_player_dist_score = 0.25 * player_dist_func(features["next_player_dist"].values)
    kick_dist_score = 0.25 * kick_dist_func(features["kick_dist"].values)
    return ball_accel_score + closest_dist_score + next_player_dist_score + kick_dist_score


# Scoring function for ETSY
max_dist = np.sqrt(PITCH_X**2 + PITCH_Y**2)
etsy_dist_func = linear_scoring_func(0, max_dist, increasing=False)


def etsy_score(features: pd.DataFrame) -> np.ndarray:
    player_ball_dist_score = 1 / 3 * etsy_dist_func(features["player_ball_dist"].values)
    player_event_dist_score = 1 / 3 * etsy_dist_func(features["player_event_dist"].values)
    ball_event_dist_score = 1 / 3 * etsy_dist_func(features["ball_event_dist"].values)
    return player_ball_dist_score + player_event_dist_score + ball_event_dist_score


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
    pass_like_types = set(PASS_LIKE_OPEN + SET_PIECE)
    direct_receive_types = set(INCOMING + ["shot_block", "keeper_punch", "bad_touch", "tackle"])
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
        if row["spadl_type"] in EVENT_END:
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

        new_row = {
            "period_id": row["period_id"],
            "episode_id": row["episode_id"] if "episode_id" in row.index else np.nan,
            "player_id": row["player_id"],
            "spadl_type": row["spadl_type"],
            "frame_id": row["frame_id"],
            "synced_ts": row[time_col] if time_col is not None else np.nan,
            "utc_timestamp": row["utc_timestamp"] if "utc_timestamp" in row.index else np.nan,
            "receiver_id": np.nan,
            "receive_frame_id": np.nan,
            "receive_ts": np.nan,
            "success": row["success"],
            "offside": row["offside"] if "offside" in row.index else False,
            "expected_goal": row["expected_goal"] if "expected_goal" in row.index else np.nan,
        }
        collapsed_rows.append(new_row)

    collapsed = pd.DataFrame(collapsed_rows)

    if collapsed.empty:
        return collapsed.reset_index(drop=True)

    for period in collapsed["period_id"].unique():
        pmask = collapsed["period_id"] == period
        collapsed.loc[pmask, "next_player_id"] = collapsed.loc[pmask, "player_id"].shift(-1)
        collapsed.loc[pmask, "next_type"] = collapsed.loc[pmask, "spadl_type"].shift(-1)

    collapsed["object_id"] = collapsed["player_id"]

    if tracking is not None:
        ball_tracking = tracking[tracking["ball"]].drop_duplicates("frame_id").set_index("frame_id")
        collapsed["start_x"] = collapsed["frame_id"].map(ball_tracking["x"])
        collapsed["start_y"] = collapsed["frame_id"].map(ball_tracking["y"])
        collapsed["end_x"] = collapsed["receive_frame_id"].map(ball_tracking["x"])
        collapsed["end_y"] = collapsed["receive_frame_id"].map(ball_tracking["y"])

    return collapsed.reset_index(drop=True)
