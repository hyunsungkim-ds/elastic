import fnmatch
from abc import ABC, abstractmethod
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class MatchData(ABC):
    def __init__(self):
        self.lineup: pd.DataFrame
        self.events: pd.DataFrame
        self.tracking: pd.DataFrame
        self.fps: float

    @staticmethod
    def calculate_event_seconds(events: pd.DataFrame) -> pd.DataFrame:
        assert "utc_timestamp" in events.columns  # in datetime
        events = events.copy()
        events["timestamp"] = 0.0

        for i in events["period_id"].unique():
            period_events: pd.DataFrame = events[events["period_id"] == i].copy()
            start_dt = period_events["utc_timestamp"].iloc[0]
            period_tds = period_events["utc_timestamp"] - start_dt
            events.loc[period_events.index, "timestamp"] = period_tds.dt.total_seconds()
            # events.loc[period_events.index, "utc_timestamp"] -= timedelta(microseconds=start_dt.microsecond)

        return events

    @staticmethod
    def calculate_tracking_datetimes(events: pd.DataFrame, tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
        assert "timestamp" in tracking.columns  # in seconds
        tracking = tracking.copy()

        if "frame_id" not in tracking.columns:
            tracking["frame_id"] = (tracking["timestamp"] * fps).round().astype(int)
            max_frame_p1 = tracking.loc[tracking["period_id"] == 1, "frame_id"].max()
            tracking.loc[tracking["period_id"] == 2, "frame_id"] += max_frame_p1 + 1

        def utc_timestamp(t: float, offset: np.datetime64) -> np.datetime64:
            return offset + timedelta(seconds=t)

        if events is not None:
            tracking["utc_timestamp"] = pd.NaT
            for i in events["period_id"].unique():
                offset = events[events["period_id"] == i]["utc_timestamp"].iloc[0]
                period_tracking = tracking[tracking["period_id"] == i]
                period_ts = period_tracking["timestamp"].apply(utc_timestamp, args=(offset,))
                tracking.loc[period_ts.index, "utc_timestamp"] = period_ts

        return tracking

    @abstractmethod
    def format_events_for_syncer(self) -> pd.DataFrame:
        pass

    def format_tracking_for_syncer(self) -> pd.DataFrame:
        tracking = self.tracking.copy()

        if "frame_id" not in tracking.columns or "utc_timestamp" not in tracking.columns:
            tracking = MatchData.calculate_tracking_datetimes(self.events, tracking, self.fps)

        home_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "home_*_x")]
        away_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "away_*_x")]
        objects = home_players + away_players + ["ball"]
        tracking_list = []

        for p in objects:
            object_tracking = tracking[["frame_id", "period_id", "timestamp", "utc_timestamp", "ball_state"]].copy()

            if p == "ball":
                object_tracking["player_id"] = None
                object_tracking["ball"] = True
            else:
                object_tracking["player_id"] = p
                object_tracking["ball"] = False

            object_tracking["x"] = tracking[f"{p}_x"].values.round(2)
            object_tracking["y"] = tracking[f"{p}_y"].values.round(2)
            object_tracking["z"] = tracking["ball_z"].values.round(2) if p == "ball" else np.nan

            for i in object_tracking["period_id"].unique():
                period_tracking = object_tracking[object_tracking["period_id"] == i].dropna(subset=["x"]).copy()
                if not period_tracking.empty:
                    vx = savgol_filter(np.diff(period_tracking["x"].values) * self.fps, window_length=15, polyorder=2)
                    vy = savgol_filter(np.diff(period_tracking["y"].values) * self.fps, window_length=15, polyorder=2)
                    speed = np.sqrt(vx**2 + vy**2)
                    period_tracking.loc[period_tracking.index[1:], "speed"] = speed
                    period_tracking["speed"] = period_tracking["speed"].bfill()

                    accel = savgol_filter(np.diff(speed) * self.fps, window_length=9, polyorder=2)
                    period_tracking.loc[period_tracking.index[1:-1], "accel_s"] = accel
                    period_tracking["accel_s"] = period_tracking["accel_s"].bfill().ffill()

                    ax = savgol_filter(np.diff(vx) * self.fps, window_length=9, polyorder=2)
                    ay = savgol_filter(np.diff(vy) * self.fps, window_length=9, polyorder=2)
                    period_tracking.loc[period_tracking.index[1:-1], "accel_v"] = np.sqrt(ax**2 + ay**2)
                    period_tracking["accel_v"] = period_tracking["accel_v"].bfill().ffill()
                    tracking_list.append(period_tracking)

        out = pd.concat(tracking_list, ignore_index=True)
        out = out[out["ball_state"] == "alive"].drop("ball_state", axis=1).reset_index(drop=True)
        return out.astype({"period_id": int, "z": float})

    @staticmethod
    def merge_events_and_tracking(events: pd.DataFrame, tracking: pd.DataFrame, fps=25, ffill=False) -> pd.DataFrame:
        if "start_x" in events.columns:
            event_cols = ["period_id", "timestamp", "object_id", "spadl_type", "start_x", "start_y"]
        else:
            event_cols = ["period_id", "timestamp", "object_id", "spadl_type", "coordinates_x", "coordinates_y"]

        events = events[event_cols].copy()
        events["timestamp"] = ((events["timestamp"] * fps).round().astype(int) / fps).round(2)

        renamed_cols = ["period_id", "timestamp", "event_player", "event_type", "event_x", "event_y"]
        merged = pd.merge(tracking, events, how="left").rename(columns=dict(zip(event_cols, renamed_cols)))

        if ffill:
            merged[renamed_cols[2:]] = merged[renamed_cols[2:]].ffill()

        return merged
