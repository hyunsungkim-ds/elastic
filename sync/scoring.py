"""Defines ETSY scoring functions."""

from typing import Callable

import numpy as np
import pandas as pd

from sync.config import FIELD_LENGTH, FIELD_WIDTH


def linear_scoring_func(min_input: float, max_input: float, increasing=False) -> Callable:
    assert min_input < max_input

    def func(x: float) -> float:
        if increasing:
            return (x - min_input) / (max_input - min_input)
        else:
            return 1 - (x - min_input) / (max_input - min_input)

    return lambda x: np.maximum(0, np.minimum(1, func(x)))


max_dist = np.sqrt(FIELD_LENGTH**2 + FIELD_WIDTH**2)
player_ball_dist_func = linear_scoring_func(0, max_dist, increasing=False)
player_event_dist_func = linear_scoring_func(0, max_dist, increasing=False)
ball_event_dist_func = linear_scoring_func(0, max_dist, increasing=False)


def score_frames_etsy(
    mask_func: Callable,
    player_ball_dists: np.ndarray,
    player_event_dists: np.ndarray,
    ball_event_dists: np.ndarray,
    ball_heights: np.ndarray,
    ball_accels: np.ndarray,
    timestamps: pd.Series,
    # bodypart,
) -> np.ndarray:
    scores = np.zeros(len(player_ball_dists))
    masked_idxs = mask_func(player_ball_dists, ball_heights, ball_accels, timestamps)

    if len(masked_idxs[0]) > 0:
        player_ball_dist_score = 100 / 3 * player_ball_dist_func(player_ball_dists[masked_idxs])
        player_event_dist_score = 100 / 3 * player_event_dist_func(player_event_dists[masked_idxs])
        ball_event_dist_score = 100 / 3 * ball_event_dist_func(ball_event_dists[masked_idxs])
        scores[masked_idxs] += player_ball_dist_score + player_event_dist_score + ball_event_dist_score

    return scores


ball_accel_func = linear_scoring_func(0, 20, increasing=True)
event_dist_func = linear_scoring_func(0, 10, increasing=False)
player_dist_func = linear_scoring_func(0, 3, increasing=False)
player_dist_func_hard = linear_scoring_func(0, 10, increasing=False)
kick_dist_func = linear_scoring_func(0, 5, increasing=True)
frame_delay_func = linear_scoring_func(0, 125, increasing=False)


def score_frames_elastic(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 20 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 20 * player_dist_func(features["player_dist"].values)
    kick_dist_score = 20 * kick_dist_func(features["kick_dist"].values)
    frame_delay_score = 40 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + kick_dist_score + frame_delay_score


def score_frames_tackle(features: pd.DataFrame) -> np.ndarray:
    ball_accel_score = 20 * ball_accel_func(features["ball_accel"].values)
    player_dist_score = 20 * player_dist_func(features["player_dist"].values)
    oppo_dist_score = 20 * player_dist_func(features["oppo_dist"].values)
    frame_delay_score = 40 * frame_delay_func(features["frame_delay"].values)
    return ball_accel_score + player_dist_score + oppo_dist_score + frame_delay_score


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
