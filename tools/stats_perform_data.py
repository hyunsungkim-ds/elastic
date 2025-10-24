from fnmatch import fnmatch
from typing import Tuple

import numpy as np
import pandas as pd

from sync import config
from tools.match_data import MatchData


class StatsPerformData(MatchData):
    def __init__(self, lineup: pd.DataFrame, events: pd.DataFrame, tracking: pd.DataFrame, fps: float = 25):
        super().__init__()

        self.lineup = lineup.copy()
        self.events = events.copy()
        self.tracking = tracking.copy()
        self.fps = fps

        lineup_cols = ["contestant_name", "shirt_number", "match_name"]
        self.lineup = self.lineup[lineup_cols].copy().sort_values(lineup_cols)

        event_dtypes = {
            "period_id": int,
            "utc_timestamp": np.dtype("datetime64[ns]"),
            "player_id": object,
            "player_name": object,
            "advanced_position": object,
            "spadl_type": object,
            "start_x": float,
            "start_y": float,
            "success": bool,
            "offside": bool,
            "aerial": bool,
            "expected_goal": float,
        }
        self.events = self.events[event_dtypes.keys()].astype(event_dtypes)

        # Filter out failed tackles
        failed_tackle_mask = (self.events["spadl_type"] == "tackle") & ~self.events["success"]
        self.events = self.events[~failed_tackle_mask].copy().reset_index(drop=True)

        self.tracking = self.tracking.copy().sort_values(["period_id", "timestamp"], ignore_index=True)
        self.tracking["timestamp"] = self.tracking["timestamp"].round(2)
        self.tracking["ball_z"] = (self.tracking["ball_z"].astype(float) / 100).round(2)  # centimeters to meters
        self.fps = fps

    @staticmethod
    def find_object_ids(
        lineup: pd.DataFrame, events: pd.DataFrame, tracking: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        lineup = lineup.copy()
        events = events.copy()

        lineup["object_id"] = None
        home_id_cols = [c for c in tracking.columns if fnmatch(c, "home_*_id")]
        away_id_cols = [c for c in tracking.columns if fnmatch(c, "away_*_id")]

        for c in home_id_cols + away_id_cols:
            player_id_series = tracking[c].dropna()
            if not player_id_series.empty:
                lineup.at[player_id_series.iloc[0], "object_id"] = c[:-3]

        events["object_id"] = events["player_id"].map(lineup["object_id"].to_dict())

        return lineup, events

    @staticmethod
    def align_event_orientations(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        home_x_cols = [c for c in tracking.columns if fnmatch(c, "home_*_x")]
        away_x_cols = [c for c in tracking.columns if fnmatch(c, "away_*_x")]

        for i in tracking["period_id"].unique():
            period_events = events[events["period_id"] == i].copy()
            home_mean_x = tracking.loc[tracking["period_id"] == i, home_x_cols].mean().mean()
            away_mean_x = tracking.loc[tracking["period_id"] == i, away_x_cols].mean().mean()

            if home_mean_x < away_mean_x:  # Rotate the away team's events
                away_events = period_events[period_events["object_id"].str.startswith("away", na=False)].copy()
                events.loc[away_events.index, "start_x"] = (config.PITCH_X - away_events["start_x"]).round(2)
                events.loc[away_events.index, "start_y"] = (config.PITCH_Y - away_events["start_y"]).round(2)
            else:  # Rotate the home team's events
                home_events = period_events[period_events["object_id"].str.startswith("home", na=False)].copy()
                events.loc[home_events.index, "start_x"] = (config.PITCH_X - home_events["start_x"]).round(2)
                events.loc[home_events.index, "start_y"] = (config.PITCH_Y - home_events["start_y"]).round(2)

        return events

    def refine_events(self):
        lineup = self.lineup.copy()
        events = self.events.copy()
        tracking = self.tracking.copy()

        events = MatchData.calculate_event_seconds(events)
        lineup, events = StatsPerformData.find_object_ids(lineup, events, tracking)
        events = StatsPerformData.align_event_orientations(events, tracking)

        # Drop invalid passes to oneself
        pass_types = ["pass", "cross", "freekick_short", "freekick_crossed"] + config.SET_PIECE_OOP
        pass_mask = events["spadl_type"].isin(pass_types)
        short_mask = events["utc_timestamp"].diff().shift(-1).dt.total_seconds() < 5
        same_poss_mask = events["player_id"] == events["player_id"].shift(-1)
        same_period_mask = events["period_id"] == events["period_id"].shift(-1)

        invalid_pass_mask = pass_mask & short_mask & same_poss_mask & same_period_mask
        events = events[~invalid_pass_mask].reset_index(drop=True).copy()

        self.lineup = lineup
        self.events = events

    def format_events_for_syncer(self) -> pd.DataFrame:
        if "timestamp" not in self.events.columns or "object_id" not in self.events.columns:
            self.refine_events()

        input_cols = ["period_id", "utc_timestamp", "object_id", "spadl_type", "start_x", "start_y", "success"]
        return self.events[input_cols].rename(columns={"object_id": "player_id"}).copy()


def find_spadl_event_types(events: pd.DataFrame, sort=True) -> pd.DataFrame:
    if sort:
        events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True, inplace=True)

    events["spadl_type"] = None
    events["success"] = events["outcome"].copy()
    events["offside"] = False
    events[["cross", "penalty"]] = events[["cross", "penalty"]].astype(bool)

    # Pass-like: pass, cross
    events.loc[(events["action_type"].str.contains("pass")) & events["cross"], "spadl_type"] = "cross"
    events.loc[(events["action_type"].str.contains("pass")) & ~events["cross"], "spadl_type"] = "pass"
    events.loc[events["action_type"].str.contains("_pass"), "success"] = False
    events.loc[events["action_type"] == "offside_pass", "offside"] = True

    # Foul and set-piece: foul, freekick_{crossed|short}, corner_{crossed|short}, goalkick
    is_foul = (events["action_type"] == "free_kick") & (events["free_kick_type"].isna() | events["penalty"])
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
    events.loc[events["spadl_type"].isin(["shot", "shot_freekick", "shot_penalty"]), "success"] = False

    is_inside_center: pd.Series = (
        (events["start_x"] >= config.PITCH_X / 2 - 3)
        & (events["start_x"] <= config.PITCH_X / 2 + 3)
        & (events["start_y"] >= config.PITCH_Y / 2 - 3)
        & (events["start_y"] <= config.PITCH_Y / 2 + 3)
    )
    events.loc[
        (events["spadl_type"].isin(["shot", "shot_freekick", "shot_penalty"]))
        & (events["period_id"].shift(-1) == events["period_id"])
        & (is_inside_center.shift(-1)),
        "success",
    ] = True
    events.loc[
        (events["spadl_type"].isin(["shot", "shot_freekick"]))
        & (events["action_type"].shift(-1) == "attempted_tackle")
        & (events["period_id"].shift(-2) == events["period_id"])
        & (is_inside_center.shift(-2)),
        "success",
    ] = True

    # Duel-like: aerial, tackle, bad_touch
    is_aerial = (events["action_type"].shift(1) == "aerial") & (events["player_id"].shift(1) == events["player_id"])
    events["aerial"] = False
    events.loc[is_aerial, "aerial"] = True
    events.loc[events["action_type"] == "attempted_tackle", "spadl_type"] = "tackle"
    events.loc[events["action_type"] == "attempted_tackle", "success"] = False
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
