import fnmatch
from typing import Tuple

import numpy as np
import pandas as pd

from sync import config
from tools.data_utils import align_directions_of_play, calculate_event_timestamps
from tools.match_data import MatchData


class StatsPerformData(MatchData):
    def __init__(self, lineup, events, tracking, fps=25):
        super().__init__(lineup, events, tracking, fps)

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
            "outcome": bool,
            "offside": bool,
            "aerial": bool,
            "expected_goal": float,
        }
        self.events = self.events[event_dtypes.keys()].astype(event_dtypes)

        # Filter out failed tackles
        failed_tackle_mask = (self.events["spadl_type"] == "tackle") & ~self.events["outcome"]
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
        home_id_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "home_*_id")]
        away_id_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "away_*_id")]

        for c in home_id_cols + away_id_cols:
            player_id_series = tracking[c].dropna()
            if not player_id_series.empty:
                lineup.at[player_id_series.iloc[0], "object_id"] = c[:-3]

        events["object_id"] = events["player_id"].map(lineup["object_id"].to_dict())

        return lineup, events

    def refine_events(self):
        lineup = self.lineup.copy()
        events = self.events.copy()
        tracking = self.tracking.copy()

        events = calculate_event_timestamps(events)
        lineup, events = StatsPerformData.find_object_ids(lineup, events, tracking)
        events = align_directions_of_play(events, tracking)

        self.lineup = lineup
        self.events = events

    def format_events_for_syncer(self) -> pd.DataFrame:
        if "timestamp" not in self.events.columns or "object_id" not in self.events.columns:
            self.refine_events()

        selected_cols = ["period_id", "utc_timestamp", "object_id", "spadl_type", "start_x", "start_y", "outcome"]
        return self.events[selected_cols].copy().rename(columns={"object_id": "player_id", "outcome": "success"})


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
