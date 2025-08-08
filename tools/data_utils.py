import fnmatch
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from sync import config


def calculate_event_timestamps(events: pd.DataFrame) -> pd.DataFrame:
    assert "utc_timestamp" in events.columns  # in datetime
    events = events.copy()
    events["timestamp"] = 0.0

    for i in events["period_id"].unique():
        period_events: pd.DataFrame = events[events["period_id"] == i].copy()
        start_dt = period_events["utc_timestamp"].iloc[0].replace(microsecond=0)
        period_tds = period_events["utc_timestamp"] - start_dt
        events.loc[period_events.index, "timestamp"] = period_tds.apply(lambda x: x.total_seconds())

    return events


def calculate_tracking_utc_timestamps(events: pd.DataFrame, tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
    assert "timestamp" in tracking.columns  # in seconds
    tracking = tracking.copy()

    tracking["frame"] = (tracking["timestamp"] * fps).round().astype(int)
    max_frame_p1 = tracking.loc[tracking["period_id"] == 1, "frame"].max()
    tracking.loc[tracking["period_id"] == 2, "frame"] += max_frame_p1 + 1

    def utc_timestamp(t: float, offset: np.datetime64) -> np.datetime64:
        return offset + timedelta(seconds=t)

    if events is not None:
        tracking["utc_timestamp"] = pd.NaT
        for i in events["period_id"].unique():
            offset = events[events["period_id"] == i]["utc_timestamp"].iloc[0].replace(microsecond=0)
            period_tracking = tracking[tracking["period_id"] == i]
            period_ts = period_tracking["timestamp"].apply(utc_timestamp, args=(offset,))
            tracking.loc[period_ts.index, "utc_timestamp"] = period_ts

    return tracking


def align_directions_of_play(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    home_x_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "home_*_x")]
    away_x_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "away_*_x")]

    for i in events["period_id"].unique():
        period_events = events[events["period_id"] == i].copy()
        home_mean_x = tracking.loc[tracking["period_id"] == i, home_x_cols].mean().mean()
        away_mean_x = tracking.loc[tracking["period_id"] == i, away_x_cols].mean().mean()

        if home_mean_x < away_mean_x:  # Rotate the away team's events
            away_events = period_events[period_events["object_id"].str.startswith("away", na=False)].copy()
            events.loc[away_events.index, "start_x"] = (config.FIELD_LENGTH - away_events["start_x"]).round(2)
            events.loc[away_events.index, "start_y"] = (config.FIELD_WIDTH - away_events["start_y"]).round(2)
        else:  # Rotate the home team's events
            home_events = period_events[period_events["object_id"].str.startswith("home", na=False)].copy()
            events.loc[home_events.index, "start_x"] = (config.FIELD_LENGTH - home_events["start_x"]).round(2)
            events.loc[home_events.index, "start_y"] = (config.FIELD_WIDTH - home_events["start_y"]).round(2)

    return events


def merge_events_and_tracking(events: pd.DataFrame, tracking: pd.DataFrame, fps=25, ffill=False) -> pd.DataFrame:
    event_cols = ["period_id", "timestamp", "object_id", "spadl_type", "start_x", "start_y"]
    renamed_cols = ["period_id", "timestamp", "event_player", "event_type", "event_x", "event_y"]

    events = events[event_cols].copy()
    events["timestamp"] = ((events["timestamp"] * fps).round().astype(int) / fps).round(2)
    merged_df = pd.merge(tracking, events, how="left").rename(columns=dict(zip(event_cols, renamed_cols)))

    if ffill:
        merged_df[renamed_cols[2:]] = merged_df[renamed_cols[2:]].ffill()

    return merged_df
