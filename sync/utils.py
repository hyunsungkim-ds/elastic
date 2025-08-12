from typing import Callable

import numpy as np
import pandas as pd

from sync.config import PITCH_X, PITCH_Y


def seconds_to_timestamp(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{int(seconds):02d}{f'{seconds % 1:.2f}'[1:]}"


def timestamp_to_seconds(timestamp: str) -> float:
    minutes, seconds = timestamp.split(":")
    return float(minutes) * 60 + float(seconds)


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
player_speed_func = linear_scoring_func(0, 5, increasing=True)
player_accel_func = linear_scoring_func(0, 5, increasing=True)
ball_accel_func = linear_scoring_func(0, 20, increasing=True)
kick_dist_func = linear_scoring_func(0, 5, increasing=True)
angle_change_func = linear_scoring_func(-1, 1, increasing=False)  # increasing from 0 to pi in radian
frame_delay_func = linear_scoring_func(0, 125, increasing=False)


def score_frames_major(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 25 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 25 * player_dist_func(features["player_dist"].values)
    kick_dist_score = 25 * kick_dist_func(features["kick_dist"].values)
    frame_delay_score = 25 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + kick_dist_score + frame_delay_score


def score_frames_tackle(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 20 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 20 * player_dist_func(features["player_dist"].values)
    oppo_dist_score = 20 * player_dist_func(features["oppo_dist"].values)
    kick_dist_score = 20 * kick_dist_func(features["kick_dist"].values)
    frame_delay_score = 20 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score + frame_delay_score


def score_frames_take_on(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 20 * ball_accel_func(features["ball_accel"].values * 2)
    max_speed_score = 20 * player_speed_func(features["max_speed"].values)
    delta_speed_score = 20 * player_speed_func(features["delta_speed"].values * 2)
    oppo_dist_score = 20 * player_dist_func(features["oppo_dist"].values - 3)
    angle_change_score = 20 * angle_change_func(features["angle_change"].values)
    return ball_accel_score + max_speed_score + delta_speed_score + oppo_dist_score + angle_change_score


def score_frames_dispossessed(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 100 / 3 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 100 / 3 * player_dist_func(features["player_dist"].values)
    kick_dist_score = 100 / 3 * kick_dist_func(features["kick_dist"].values)
    return ball_accel_score + player_dist_score + kick_dist_score


def score_frames_receive(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 25 * ball_accel_func(features["ball_accel"].values)
    closest_dist_score = 25 * player_dist_func(features["closest_dist"].values)
    next_player_dist_score = 25 * player_dist_func(features["next_player_dist"].values)
    kick_dist_score = 25 * kick_dist_func(features["kick_dist"].values)
    return ball_accel_score + closest_dist_score + next_player_dist_score + kick_dist_score


# Scoring function for ETSY
max_dist = np.sqrt(PITCH_X**2 + PITCH_Y**2)
etsy_dist_func = linear_scoring_func(0, max_dist, increasing=False)


def score_frames_etsy(features: pd.DataFrame) -> np.ndarray:
    player_ball_dist_score = 100 / 3 * etsy_dist_func(features["player_ball_dist"].values)
    player_event_dist_score = 100 / 3 * etsy_dist_func(features["player_event_dist"].values)
    ball_event_dist_score = 100 / 3 * etsy_dist_func(features["ball_event_dist"].values)
    return player_ball_dist_score + player_event_dist_score + ball_event_dist_score
