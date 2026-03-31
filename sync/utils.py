from datetime import datetime, timedelta
from typing import Callable, Union

import numpy as np
import pandas as pd

from sync.config import PITCH_X, PITCH_Y


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
player_dist_func = linear_scoring_func(1, 3, increasing=False)
out_dist_func = linear_scoring_func(0, 1, increasing=False)
player_speed_func = linear_scoring_func(0, 5, increasing=True)
player_accel_func = linear_scoring_func(0, 5, increasing=True)
ball_accel_func = linear_scoring_func(0, 30, increasing=True)
kick_dist_func = linear_scoring_func(0, 5, increasing=True)
angle_change_func = linear_scoring_func(0, 1, increasing=False)  # increasing from 0 to pi in radian
frame_delay_func = linear_scoring_func(0, 125, increasing=False)
positive_slope_penalty = linear_scoring_func(0, 0.2, increasing=False)
negative_slope_penalty = linear_scoring_func(-0.2, 0, increasing=True)


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
        player_dist_score = 30 * dist_func(features["player_dist"].to_numpy())
        player_dist_slope_score = 20 * slope_func(features[slope_col].to_numpy())
        kick_dist_score = 30 * kick_dist_func(features[kick_dist_col].to_numpy())
        ball_accel_score = 20 * ball_accel_func(features["ball_accel"].to_numpy())
        scores[mask.to_numpy()] = player_dist_score + player_dist_slope_score + kick_dist_score + ball_accel_score
        return scores

    elif features["player_id"] == player_id:  # isinstance(features, pd.Series)
        player_dist_score = 30 * dist_func(features["player_dist"])
        player_dist_slope_score = 20 * slope_func(features[slope_col])
        kick_dist_score = 30 * kick_dist_func(features[kick_dist_col])
        ball_accel_score = 20 * ball_accel_func(features["ball_accel"])
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
        ball_accel_score = 25 * ball_accel_func(features["ball_accel"].to_numpy())
        player_dist_score = 25 * player_dist_func(features["player_dist"].to_numpy())
        oppo_dist_score = 25 * player_dist_func(features["oppo_dist"].to_numpy())
        kick_dist_score = 25 * kick_dist_func(features[kick_dist_col].to_numpy())
        scores[mask.to_numpy()] = ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score
        return scores

    elif features["player_id"] == player_id:  # isinstance(features, pd.Series)
        ball_accel_score = 25 * ball_accel_func(features["ball_accel"])
        player_dist_score = 25 * player_dist_func(features["player_dist"])
        oppo_dist_score = 25 * player_dist_func(features["oppo_dist"])
        kick_dist_score = 25 * kick_dist_func(features[kick_dist_col])
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
        20 * ball_accel_func(f["ball_accel"].to_numpy() * 2)
        + 20 * player_speed_func(f["max_speed"].fillna(0).to_numpy())
        # + 20 * player_speed_func(f["delta_speed"].fillna(0).to_numpy() * 2)
        + 20 * player_dist_func(f["oppo_dist"].fillna(10).to_numpy() - 3)
        + 40 * angle_change_func(f["angle_change"].fillna(0).to_numpy())
    )
    return scores


def greedy_score_major(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 25 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 25 * player_dist_func(features["player_dist"].values)
    kick_dist_score = 25 * kick_dist_func(features["kick_dist"].values)
    frame_delay_score = 25 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + kick_dist_score + frame_delay_score


def greedy_score_tackle(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 20 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 20 * player_dist_func(features["player_dist"].values)
    oppo_dist_score = 20 * player_dist_func(features["oppo_dist"].values)
    kick_dist_score = 20 * kick_dist_func(features["kick_dist"].values)
    frame_delay_score = 20 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score + frame_delay_score


def greedy_score_takeon(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 20 * ball_accel_func(features["ball_accel"].values * 2)
    max_speed_score = 20 * player_speed_func(features["max_speed"].values)
    delta_speed_score = 20 * player_speed_func(features["delta_speed"].values * 2)
    oppo_dist_score = 20 * player_dist_func(features["oppo_dist"].values - 3)
    angle_change_score = 20 * angle_change_func(features["angle_change"].values)
    return ball_accel_score + max_speed_score + delta_speed_score + oppo_dist_score + angle_change_score


def greedy_score_dispossessed(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 100 / 3 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 100 / 3 * player_dist_func(features["player_dist"].values)
    kick_dist_score = 100 / 3 * kick_dist_func(features["kick_dist"].values)
    return ball_accel_score + player_dist_score + kick_dist_score


def greedy_score_receive(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 25 * ball_accel_func(features["ball_accel"].values)
    closest_dist_score = 25 * player_dist_func(features["closest_dist"].values)
    next_player_dist_score = 25 * player_dist_func(features["next_player_dist"].values)
    kick_dist_score = 25 * kick_dist_func(features["kick_dist"].values)
    return ball_accel_score + closest_dist_score + next_player_dist_score + kick_dist_score


# Scoring function for ETSY
max_dist = np.sqrt(PITCH_X**2 + PITCH_Y**2)
etsy_dist_func = linear_scoring_func(0, max_dist, increasing=False)


def etsy_score(features: pd.DataFrame) -> np.ndarray:
    player_ball_dist_score = 100 / 3 * etsy_dist_func(features["player_ball_dist"].values)
    player_event_dist_score = 100 / 3 * etsy_dist_func(features["player_event_dist"].values)
    ball_event_dist_score = 100 / 3 * etsy_dist_func(features["ball_event_dist"].values)
    return player_ball_dist_score + player_event_dist_score + ball_event_dist_score
