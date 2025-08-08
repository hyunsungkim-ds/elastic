import fnmatch
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from tools.data_utils import calculate_tracking_utc_timestamps


class MatchData(ABC):
    def __init__(self, lineup: pd.DataFrame, events: pd.DataFrame, tracking: pd.DataFrame, fps=25):
        self.lineup = lineup.copy()
        self.events = events.copy()
        self.tracking = tracking.copy()
        self.fps = fps

    @abstractmethod
    def format_events_for_syncer(self) -> pd.DataFrame:
        pass

    def format_tracking_for_syncer(self) -> pd.DataFrame:
        tracking = self.tracking.copy()

        if "frame" not in tracking.columns or "utc_timestamp" not in tracking.columns:
            tracking = calculate_tracking_utc_timestamps(self.events, tracking, self.fps)

        home_players = [c[:-3] for c in tracking.columns if fnmatch.fnmatch(c, "home_*_id")]
        away_players = [c[:-3] for c in tracking.columns if fnmatch.fnmatch(c, "away_*_id")]
        objects = home_players + away_players + ["ball"]
        tracking_list = []

        for p in objects:
            object_tracking = tracking[["frame", "period_id", "timestamp", "utc_timestamp", "ball_state"]].copy()

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
