import fnmatch
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from sync import config


class Preprocessor:
    def __init__(self, lineup: pd.DataFrame, events: pd.DataFrame, traces: pd.DataFrame, fps=25):
        lineup_cols = ["contestant_name", "shirt_number", "match_name"]
        self.lineup = lineup[lineup_cols].copy().sort_values(lineup_cols)

        event_dtypes = {
            "period_id": int,
            "utc_timestamp": np.dtype("datetime64[ns]"),
            "player_id": object,
            "player_name": object,
            "advanced_position": object,
            "spadl_type": object,
            "start_x": float,
            "start_y": float,
            "outcome": bool,
            "offside": bool,
            "aerial": bool,
            "expected_goal": float,
        }
        events = events[event_dtypes.keys()].astype(event_dtypes)
        self.events = events[(events["spadl_type"] != "tackle") | events["outcome"]].copy().reset_index(drop=True)

        self.traces = traces.copy().sort_values(["period_id", "timestamp"], ignore_index=True)
        self.traces["timestamp"] = self.traces["timestamp"].round(2)
        self.traces["ball_z"] = (self.traces["ball_z"].astype(float) / 100).round(2)  # centimeters to meters
        self.fps = fps

    def calculate_event_timestamps(self):
        self.events["timestamp"] = 0.0
        for i in self.events["period_id"].unique():
            period_events = self.events[self.events["period_id"] == i].copy()
            period_tds = period_events["utc_timestamp"] - period_events["utc_timestamp"].iloc[0].replace(microsecond=0)
            self.events.loc[period_events.index, "timestamp"] = period_tds.apply(lambda x: x.total_seconds())

    def calculate_tracking_utc_timestamps(self):
        self.traces["frame"] = (self.traces["timestamp"] * self.fps).round().astype(int)
        max_frame_p1 = self.traces.loc[self.traces["period_id"] == 1, "frame"].max()
        self.traces.loc[self.traces["period_id"] == 2, "frame"] += max_frame_p1 + 1

        def utc_timestamp(t: float, offset: np.datetime64) -> np.datetime64:
            return offset + timedelta(seconds=t)

        if self.events is not None:
            self.traces["utc_timestamp"] = pd.NaT
            for i in [1, 2]:
                offset = self.events[self.events["period_id"] == i]["utc_timestamp"].iloc[0].replace(microsecond=0)
                period_ts = self.traces[self.traces["period_id"] == i]["timestamp"].apply(utc_timestamp, args=(offset,))
                self.traces.loc[period_ts.index, "utc_timestamp"] = period_ts

    def find_object_ids(self):
        self.lineup["object_id"] = None
        home_id_cols = [c for c in self.traces.columns if fnmatch.fnmatch(c, "home_*_id")]
        away_id_cols = [c for c in self.traces.columns if fnmatch.fnmatch(c, "away_*_id")]

        for c in home_id_cols + away_id_cols:
            player_id_series = self.traces[c].dropna()
            if not player_id_series.empty:
                self.lineup.at[player_id_series.iloc[0], "object_id"] = c[:-3]

        self.events["object_id"] = self.events["player_id"].map(self.lineup["object_id"].to_dict())

    def align_directions_of_play(self):
        home_x_cols = [c for c in self.traces.columns if fnmatch.fnmatch(c, "home_*_x")]
        away_x_cols = [c for c in self.traces.columns if fnmatch.fnmatch(c, "away_*_x")]

        for i in self.events["period_id"].unique():
            period_events = self.events[self.events["period_id"] == i].copy()
            home_mean_x = self.traces.loc[self.traces["period_id"] == i, home_x_cols].mean().mean()
            away_mean_x = self.traces.loc[self.traces["period_id"] == i, away_x_cols].mean().mean()

            if home_mean_x < away_mean_x:  # Rotate the away team's events
                away_events = period_events[period_events["object_id"].str.startswith("away", na=False)].copy()
                self.events.loc[away_events.index, "start_x"] = (105 - away_events["start_x"]).round(2)
                self.events.loc[away_events.index, "start_y"] = (68 - away_events["start_y"]).round(2)
            else:  # Rotate the home team's events
                home_events = period_events[period_events["object_id"].str.startswith("home", na=False)].copy()
                self.events.loc[home_events.index, "start_x"] = (105 - home_events["start_x"]).round(2)
                self.events.loc[home_events.index, "start_y"] = (68 - home_events["start_y"]).round(2)

    def refine_events(self):
        self.calculate_event_timestamps()
        self.find_object_ids()
        self.align_directions_of_play()

    def combine_events_and_traces(self, ffill=False) -> pd.DataFrame:
        merged_cols = ["period_id", "timestamp", "object_id", "spadl_type", "start_x", "start_y"]
        renamed_cols = ["period_id", "timestamp", "event_player", "event_type", "event_x", "event_y"]

        events = self.events[merged_cols].copy()
        events["timestamp"] = ((events["timestamp"] * self.fps).round().astype(int) / self.fps).round(2)
        ret = pd.merge(self.traces, events, how="left").rename(columns=dict(zip(merged_cols, renamed_cols)))

        if ffill:
            ret[renamed_cols[2:]] = ret[renamed_cols[2:]].ffill()

        return ret

    def format_events_for_syncer(self) -> pd.DataFrame:
        if "timestamp" not in self.events.columns or "object_id" not in self.events.columns:
            self.refine_events()

        cols = ["period_id", "utc_timestamp", "object_id", "spadl_type", "start_x", "start_y", "outcome", "offside"]
        return self.events[cols].copy().rename(columns={"object_id": "player_id"})

    def format_traces_for_syncer(self) -> pd.DataFrame:
        if "frame" not in self.traces.columns or "utc_timestamp" not in self.traces.columns:
            self.calculate_tracking_utc_timestamps()

        traces = self.traces[self.traces["ball_state"] == "alive"].copy()

        home_players = [c[:-3] for c in traces.columns if fnmatch.fnmatch(c, "home_*_id")]
        away_players = [c[:-3] for c in traces.columns if fnmatch.fnmatch(c, "away_*_id")]
        objects = home_players + away_players + ["ball"]
        ret_list = []

        for p in objects:
            object_traces = traces[["frame", "period_id", "timestamp", "utc_timestamp"]].copy()

            if p == "ball":
                object_traces["player_id"] = None
                object_traces["ball"] = True
            else:
                object_traces["player_id"] = p  # traces[f"{p}_id"]
                object_traces["ball"] = False

            object_traces["x"] = traces[f"{p}_x"].round(2)
            object_traces["y"] = traces[f"{p}_y"].round(2)
            object_traces["z"] = traces["ball_z"].round(2) if p == "ball" else np.nan

            for i in object_traces["period_id"].unique():
                period_traces = object_traces[object_traces["period_id"] == i].dropna(subset=["x"]).copy()
                if not period_traces.empty:
                    vx = savgol_filter(np.diff(period_traces["x"].values) * self.fps, window_length=15, polyorder=2)
                    vy = savgol_filter(np.diff(period_traces["y"].values) * self.fps, window_length=15, polyorder=2)
                    # speed = np.sqrt(vx**2 + vy**2)
                    # accel = savgol_filter(np.diff(speed) * self.fps, window_length=9, polyorder=2)
                    # period_traces.loc[period_traces.index[1:-1], "accel"] = accel
                    # period_traces["accel"] = period_traces["accel"].bfill().ffill()
                    ax = savgol_filter(np.diff(vx) * self.fps, window_length=9, polyorder=2)
                    ay = savgol_filter(np.diff(vy) * self.fps, window_length=9, polyorder=2)
                    period_traces.loc[period_traces.index[1:-1], "accel"] = np.sqrt(ax**2 + ay**2)
                    period_traces["accel"] = period_traces["accel"].bfill().ffill()
                    ret_list.append(period_traces)

        ret = pd.concat(ret_list, ignore_index=True)
        return ret.astype({"period_id": int, "z": float})


def find_spadl_event_types(events: pd.DataFrame, sort=True) -> pd.DataFrame:
    if sort:
        events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True, inplace=True)

    events["spadl_type"] = None
    events["offside"] = False
    events[["cross", "penalty"]] = events[["cross", "penalty"]].astype(bool)

    # Pass-like: pass, cross
    events.loc[(events["action_type"].str.contains("pass")) & events["cross"], "spadl_type"] = "cross"
    events.loc[(events["action_type"].str.contains("pass")) & ~events["cross"], "spadl_type"] = "pass"
    events.loc[events["action_type"].str.contains("_pass"), "outcome"] = False
    events.loc[events["action_type"] == "offside_pass", "offside"] = True

    # Foul and set-piece: foul, freekick_{crossed|short}, corner_{crossed|short}, goalkick
    is_foul = (events["action_type"] == "free_kick") & (events["free_kick_type"].isna() | events["penalty"])
    # events.loc[is_foul & ~events["outcome"], "spadl_type"] = "foul"
    events.loc[is_foul, "spadl_type"] = "foul"

    is_freekick = (events["action_type"] == "free_kick") & events["free_kick_type"].notna()
    events.loc[is_freekick & events["expected_goal"].isna() & events["cross"], "spadl_type"] = "freekick_crossed"
    events.loc[is_freekick & events["expected_goal"].isna() & ~events["cross"], "spadl_type"] = "freekick_short"

    events.loc[(events["action_type"] == "corner") & events["cross"], "spadl_type"] = "corner_crossed"
    events.loc[(events["action_type"] == "corner") & ~events["cross"], "spadl_type"] = "corner_short"
    events.loc[events["action_type"] == "goal_kick", "spadl_type"] = "goalkick"

    # Shot-like: shot, shot_freekick, shot_penalty
    events.loc[(events["action_type"] == "goal_attempt") & ~events["penalty"], "spadl_type"] = "shot"
    events.loc[(events["action_type"] == "goal_attempt") & events["penalty"], "spadl_type"] = "shot_penalty"
    events.loc[is_freekick & events["expected_goal"].notna(), "spadl_type"] = "shot_freekick"
    events.loc[events["spadl_type"].isin(["shot", "shot_freekick", "shot_penalty"]), "outcome"] = False

    is_inside_center: pd.Series = (
        (events["start_x"] >= config.FIELD_LENGTH / 2 - 3)
        & (events["start_x"] <= config.FIELD_LENGTH / 2 + 3)
        & (events["start_y"] >= config.FIELD_WIDTH / 2 - 3)
        & (events["start_y"] <= config.FIELD_WIDTH / 2 + 3)
    )
    events.loc[
        (events["spadl_type"].isin(["shot", "shot_freekick", "shot_penalty"]))
        & (events["period_id"].shift(-1) == events["period_id"])
        & (is_inside_center.shift(-1)),
        "outcome",
    ] = True
    events.loc[
        (events["spadl_type"].isin(["shot", "shot_freekick"]))
        & (events["action_type"].shift(-1) == "attempted_tackle")
        & (events["period_id"].shift(-2) == events["period_id"])
        & (is_inside_center.shift(-2)),
        "outcome",
    ] = True

    # Duel-like: aerial, tackle, bad_touch
    is_aerial = (events["action_type"].shift(1) == "aerial") & (events["player_id"].shift(1) == events["player_id"])
    events["aerial"] = False
    events.loc[is_aerial, "aerial"] = True
    events.loc[events["action_type"] == "attempted_tackle", "spadl_type"] = "tackle"
    events.loc[events["action_type"] == "attempted_tackle", "outcome"] = False
    events.loc[events["action_type"] == "ball_touch", "spadl_type"] = "bad_touch"

    # Keeper actions: keeper_{save|claim|punch|pick_up|sweeper}
    is_save = events["action_type"] == "save"
    events.loc[is_save & (events["advanced_position"] == "goal_keeper"), "spadl_type"] = "keeper_save"
    events.loc[is_save & (events["advanced_position"] != "goal_keeper"), "spadl_type"] = "shot_block"

    events.loc[events["action_type"] == "claim", "spadl_type"] = "keeper_claim"
    events.loc[events["action_type"] == "punch", "spadl_type"] = "keeper_punch"
    events.loc[events["action_type"] == "keeper_pick-up", "spadl_type"] = "keeper_pick_up"

    is_ks = events["action_type"] == "keeper_sweeper"
    events.loc[is_ks & (events["advanced_position"].shift(-1) == "goal_keeper"), "spadl_type"] = "keeper_sweeper"
    events.loc[is_ks & (events["advanced_position"].shift(-1) != "goal_keeper"), "spadl_type"] = "clearance"

    # Types to maintain their names
    types_as_is = [
        "throw_in",
        "take_on",
        "tackle",
        "interception",
        "clearance",
        "bad_touch",
        "ball_recovery",
        "dispossessed",
        "foul",
    ]
    events_as_is = events[events["action_type"].isin(types_as_is)]
    events.loc[events_as_is.index, "spadl_type"] = events_as_is["action_type"]

    return events
