import os
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from kloppy import sportec
from kloppy.domain import Dimension, MetricPitchDimensions, Orientation, TrackingDataset
from scipy.signal import savgol_filter

from sync import config
from tools.match_data import MatchData

META_DIR = "data/sportec/metadata"
EVENT_DIR = "data/sportec/event"
TRACKING_DIR = "data/sportec/tracking"

POSITION_MAPPING = {
    None: None,
    "TW": "GK",
    "IVR": "RCB",
    "IVL": "LCB",
    "IVZ": "CB",
    "RV": "RB",
    "LV": "LB",
    "DMR": "RDM",
    "DRM": "RDM",
    "DML": "LDM",
    "DLM": "LDM",
    "DMZ": "CDM",
    "HR": "RCM",
    "HL": "LCM",
    "MZ": "CM",
    "RM": "RM",
    "LM": "LM",
    "ORM": "RAM",
    "OLM": "LAM",
    "ZO": "CAM",
    "RA": "RWF",
    "LA": "LWF",
    "STR": "RCF",
    "STL": "LCF",
    "STZ": "CF",
}


class SportecData(MatchData):
    def __init__(self, match_id: str):
        self.match_id = match_id

        meta_files = [f for f in os.listdir(META_DIR) if "matchinformation" in f and match_id in f]
        event_files = [f for f in os.listdir(EVENT_DIR) if "events" in f and match_id in f]
        tracking_files = [f for f in os.listdir(TRACKING_DIR) if "positions" in f and match_id in f]

        assert meta_files and event_files and tracking_files

        self.meta_path = f"{META_DIR}/{meta_files[0]}"
        self.event_path = f"{EVENT_DIR}/{event_files[0]}"
        self.tracking_path = f"{TRACKING_DIR}/{tracking_files[0]}"

        lineup = self.load_lineup_data(self.meta_path)
        events = self.load_event_data(self.event_path)

        # Delay loading tracking data since it often takes more than a minute
        super().__init__(lineup, events, tracking=None)

    @staticmethod
    def load_lineup_data(meta_path: str) -> pd.DataFrame:
        tree = ET.parse(meta_path)
        root = tree.getroot()
        lineup_list = []

        for team in root.findall(".//Team"):
            team_id = team.attrib.get("TeamId")
            team_name = team.attrib.get("TeamName")
            home_away = "away" if team.attrib.get("Role") == "guest" else "home"

            players = team.find("Players")
            if players is not None:
                for player in players.findall("Player"):
                    uniform_number = int(player.attrib.get("ShirtNumber"))
                    lineup_list.append(
                        {
                            "team_id": team_id,
                            "team_name": team_name,
                            "home_away": home_away,
                            "player_id": player.attrib.get("PersonId"),
                            "uniform_number": uniform_number,
                            "object_id": f"{home_away}_{uniform_number}",
                            "player_name": player.attrib.get("Shortname"),
                            "starting": player.attrib.get("Starting") == "true",
                            "playing_position": POSITION_MAPPING[player.attrib.get("PlayingPosition")],
                            "captain": player.attrib.get("TeamLeader") == "true",
                        }
                    )

        return pd.DataFrame(lineup_list).sort_values(["home_away", "uniform_number"], ignore_index=True)

    @staticmethod
    def load_event_data(event_path: str) -> pd.DataFrame:
        def parse_play(play: Optional[ET.Element] = None) -> Tuple[str, str, str, str, str, bool, str]:
            if play is not None and play.tag == "Play":
                if play.find("Pass") is not None:
                    event_type = "Pass"
                elif play.find("Cross") is not None:
                    event_type = "Cross"
                else:
                    event_type = "Play"

                team_id = play.attrib.get("Team")
                player_id = play.attrib.get("Player")
                receiver_id = play.attrib.get("Recipient")
                result = play.attrib.get("Evaluation")
                success = result == "successfullyCompleted"
                body_part = None

                return event_type, team_id, player_id, receiver_id, result, success, body_part

            elif play is not None and play.tag == "ShotAtGoal":
                event_type = "Shot"
                team_id = play.attrib.get("Team")
                player_id = play.attrib.get("Player")
                receiver_id = None
                result = None
                success = True if play.find("SuccessfulShot") else False

                if play.find("ShotWide") is not None:
                    result = "OffTarget"
                elif play.find("SavedShot") is not None:
                    receiver_id = play.find("SavedShot").attrib.get("GoalKeeper")
                    result = "Saved"
                elif play.find("BlockedShot") is not None:
                    receiver_id = play.find("BlockedShot").attrib.get("Player")
                    result = "Blocked"
                elif play.find("ShotWoodWork") is not None:
                    result = "Post"
                elif play.find("SuccessfulShot") is not None:
                    result = "Goal"

                if play.attrib.get("TypeOfShot") == "head":
                    body_part = "Head"
                elif play.attrib.get("TypeOfShot") == "leftLeg":
                    body_part = "LeftFoot"
                elif play.attrib.get("TypeOfShot") == "rightLeg":
                    body_part = "RightFoot"
                else:
                    body_part = None

                return event_type, team_id, player_id, receiver_id, result, success, body_part

            else:
                return None, None, None, None, None, None, None

        tree = ET.parse(event_path)
        root = tree.getroot()

        event_rows = []

        for event in root.findall(".//Event"):
            event_id = event.attrib.get("EventId")
            period_id = 0
            timestamp = event.attrib.get("EventTime")

            child = next((c for c in event if c.tag not in ["Qualifier"]), None)
            event_type = child.tag if child is not None else "UNKNOWN"
            if event_type == "Delete":
                continue

            team_id = child.attrib.get("Team")
            player_id = child.attrib.get("Player")

            x = event.attrib.get("X-Position") or event.attrib.get("X-Source-Position")
            y = event.attrib.get("Y-Position") or event.attrib.get("Y-Source-Position")
            end_x = event.attrib.get("End-X-Position")
            end_y = event.attrib.get("End-Y-Position")

            receiver_id = None
            set_piece_type = None
            result = None
            success = None
            body_part = None
            card_type = None

            if child.tag in ["Play", "ShotAtGoal"]:
                event_type, team_id, player_id, receiver_id, result, success, body_part = parse_play(child)

            elif child.tag in ["KickOff", "FinalWhistle"]:
                set_piece_type = child.tag
                if child.attrib.get("GameSection") == "firstHalf":
                    period_id = 1
                elif child.attrib.get("GameSection") == "secondHalf":
                    period_id = 2
                if set_piece_type == "KickOff":
                    play = child.find("Play")
                    event_type, team_id, player_id, receiver_id, result, success, body_part = parse_play(play)

            elif child.tag in ["KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty"]:
                set_piece_type = child.tag
                if child.find("Play") is not None:
                    play = child.find("Play")
                elif child.find("ShotAtGoal") is not None:
                    play = child.find("ShotAtGoal")
                else:
                    play = None
                event_type, team_id, player_id, receiver_id, result, success, body_part = parse_play(play)

            elif child.tag == "TacklingGame":
                team_id = child.attrib.get("WinnerTeam")
                player_id = child.attrib.get("Winner")
                receiver_id = child.attrib.get("Loser")
                success = True
                if child.attrib.get("PossessionChange") == "true":
                    result = "PossessionChange"
                if child.attrib.get("Type") == "air":
                    event_type = "AerialDuel"
                elif child.attrib.get("Type") == "ground":
                    event_type = "GroundDuel"

            elif child.tag == "BallClaiming":
                if child.attrib.get("Type") == "InterceptedBall":
                    event_type = "Interception"
                elif child.attrib.get("Type") == "BallClaimed":
                    event_type = "Recovery"

            elif child.tag == "OtherBallAction":
                if child.attrib.get("DefensiveClearance") == "true":
                    event_type = "Clearance"
                else:
                    event_type = "OtherBallAction"

            elif child.tag == "Foul":
                team_id = child.attrib.get("TeamFouler")
                player_id = child.attrib.get("Fouler")
                receiver_id = child.attrib.get("Fouled")
                result = child.attrib.get("FoulType")

            elif child.tag == "Caution":
                event_type = "Card"
                if child.attrib.get("CardColor") == "yellow":
                    card_type = "Yellow"
                elif child.attrib.get("CardColor") == "red":
                    card_type = "Red"

            elif child.tag == "Substitution":
                team_id = child.attrib.get("Team")
                player_id = child.attrib.get("PlayerOut")
                receiver_id = child.attrib.get("PlayerIn")

            event_rows.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "period_id": period_id,
                    "utc_timestamp": timestamp,
                    "team_id": team_id,
                    "player_id": player_id,
                    "coordinates_x": float(x) if x else None,
                    "coordinates_y": float(y) if y else None,
                    "end_coordinates_x": float(end_x) if end_x else None,
                    "end_coordinates_y": float(end_y) if end_y else None,
                    "receiver_player_id": receiver_id,
                    "set_piece_type": set_piece_type,
                    "result": result,
                    "success": success,
                    "body_part_type": body_part,
                    "card_type": card_type,
                }
            )

        events = pd.DataFrame(event_rows)
        events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
        events.sort_values("utc_timestamp", ignore_index=True, inplace=True)

        period_ids = [id for id in events["period_id"].unique() if id > 0]
        for period_id in period_ids:
            period_events = events[events["period_id"] == period_id].copy()
            start_idx = period_events[period_events["set_piece_type"] == "KickOff"].index[0]
            end_idx = period_events[period_events["event_type"] == "FinalWhistle"].index[-1]
            events.loc[start_idx:end_idx, "period_id"] = period_id

        return events

    @staticmethod
    def load_tracking_data(
        tracking_path: str, meta_path: str, lineup: pd.DataFrame
    ) -> Tuple[TrackingDataset, pd.DataFrame]:
        print("Loading the tracking data...")
        tracking_ds = sportec.load_tracking(raw_data=tracking_path, meta_data=meta_path, only_alive=False)

        print("Transforming the tracking data coordinates...")
        pitch_dims = MetricPitchDimensions(standardized=True, x_dim=Dimension(0, 105), y_dim=Dimension(0, 68))
        tracking_ds = tracking_ds.transform(
            to_orientation=Orientation.HOME_AWAY,
            to_pitch_dimensions=pitch_dims,
        )

        tracking_df: pd.DataFrame = tracking_ds.to_df()
        player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
        column_mapping = {f"{k}_{t}": f"{v}_{t}" for k, v in player_mapping.items() for t in ["x", "y", "d", "s"]}
        tracking_df = tracking_df.rename(columns=column_mapping)

        player_x_cols = [c for c in tracking_df.columns if fnmatch(c, "home_*_x") or fnmatch(c, "away_*_x")]
        tracking_df = tracking_df.dropna(subset=player_x_cols, how="all").copy()

        return tracking_ds, tracking_df

    @staticmethod
    def find_object_ids(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
        events["object_id"] = events["player_id"].map(player_mapping)
        events["receiver_id"] = events["receiver_player_id"].map(player_mapping)

        return events

    @staticmethod
    def find_spadl_event_types(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["spadl_type"] = None

        pass_mask = events["event_type"] == "Pass"
        events.loc[pass_mask, "spadl_type"] = "pass"
        events.loc[pass_mask & (events["set_piece_type"] == "ThrowIn"), "spadl_type"] = "throw_in"
        events.loc[pass_mask & (events["set_piece_type"] == "GoalKick"), "spadl_type"] = "goalkick"
        events.loc[pass_mask & (events["set_piece_type"] == "CornerKick"), "spadl_type"] = "corner_short"
        events.loc[pass_mask & (events["set_piece_type"] == "FreeKick"), "spadl_type"] = "freekick_short"

        cross_mask = events["event_type"] == "Cross"
        events.loc[cross_mask, "spadl_type"] = "cross"
        events.loc[cross_mask & (events["set_piece_type"] == "CornerKick"), "spadl_type"] = "corner_crossed"
        events.loc[cross_mask & (events["set_piece_type"] == "FreeKick"), "spadl_type"] = "freekick_crossed"

        shot_mask = events["event_type"] == "Shot"
        events.loc[shot_mask, "spadl_type"] = "shot"
        events.loc[shot_mask & (events["set_piece_type"] == "FreeKick"), "spadl_type"] = "shot_freekick"
        events.loc[shot_mask & (events["set_piece_type"] == "Penalty"), "spadl_type"] = "shot_penalty"

        events.loc[events["event_type"] == "Interception", "spadl_type"] = "interception"
        events.loc[events["event_type"] == "Recovery", "spadl_type"] = "recovery"
        events.loc[events["event_type"] == "Clearance", "spadl_type"] = "clearance"
        events.loc[events["event_type"] == "Foul", "spadl_type"] = "foul"

        for i in events[events["event_type"] == "OtherBallAction"].index:
            team_id = events.at[i, "team_id"]
            player_id = events.at[i, "player_id"]
            recent_action = events[~events["event_type"].str.contains("Duel")].loc[: i - 1].iloc[-1]

            if recent_action["receiver_player_id"] == player_id:
                if recent_action["event_type"] in ["Pass", "Cross"] and not recent_action["success"]:
                    events.at[i, "spadl_type"] = "interception"
                    continue

                elif recent_action["event_type"] == "Shot" and not recent_action["success"]:
                    events.at[i, "spadl_type"] = "shot_block"
                    continue

                # If the player is not involved in adjoining ground duels and he/she loses possession
                adj_duels = events[events["event_type"] == "GroundDuel"].loc[i - 1 : i + 2]
                if player_id not in adj_duels["player_id"].tolist() + adj_duels["receiver_player_id"].tolist():
                    if events.at[i + 1, "player_id"] != player_id:
                        events.at[i, "spadl_type"] = "bad_touch"
                        continue

            if recent_action["event_type"] == "Clearance":
                events.at[i, "spadl_type"] = "recovery"

            if events.at[i + 1, "event_type"] == "GroundDuel":
                duel_winner_id = events.at[i + 1, "player_id"]
                duel_loser_id = events.at[i + 1, "receiver_player_id"]

                # If the player is the winner of the following ground duel
                if duel_winner_id == player_id:
                    prev_player_id = events.at[i - 1, "player_id"]
                    prev_event_type = events.at[i - 1, "event_type"]

                    if prev_event_type == "OtherBallAction" and duel_loser_id == prev_player_id:
                        events.at[i - 1, "spadl_type"] = "dispossessed"
                        events.at[i, "spadl_type"] = "tackle"
                        continue

                    elif recent_action["team_id"] != team_id:
                        events.at[i, "spadl_type"] = "tackle"
                        continue

                # If the player is the loser of the following ground duel
                if duel_loser_id == player_id:
                    events.at[i, "spadl_type"] = "dispossessed"
                    continue

            if events.at[i - 1, "event_type"] == "GroundDuel":
                duel_winner_id = events.at[i - 1, "player_id"]
                duel_loser_id = events.at[i - 1, "receiver_player_id"]

                # If the player is the winner of the previous ground duel
                if duel_winner_id == player_id or duel_loser_id == player_id:
                    events.at[i, "spadl_type"] = "tackle"

        return events

    @staticmethod
    def align_directions_of_play(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        assert "object_id" in events.columns
        events = events.copy()

        home_x_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "home_*_x")]
        away_x_cols = [c for c in tracking.columns if fnmatch.fnmatch(c, "away_*_x")]

        for i in tracking["period_id"].unique():
            period_events = events[events["period_id"] == i].copy()
            home_mean_x = tracking.loc[tracking["period_id"] == i, home_x_cols].mean().mean()
            away_mean_x = tracking.loc[tracking["period_id"] == i, away_x_cols].mean().mean()

            if home_mean_x < away_mean_x:  # Flip the x-axis of the home team's events
                home_events = period_events[period_events["object_id"].str.startswith("home", na=False)].copy()
                events.loc[home_events.index, "start_x"] = (config.FIELD_LENGTH - home_events["start_x"]).round(2)
            else:  # Flip the x-axis of the away team's events
                away_events = period_events[period_events["object_id"].str.startswith("away", na=False)].copy()
                events.loc[away_events.index, "start_x"] = (config.FIELD_LENGTH - home_events["start_x"]).round(2)

        return events

    def format_events_for_syncer(self) -> pd.DataFrame:
        if self.events is None:
            self.events = SportecData.load_event_data(self.event_path)

        events = SportecData.find_object_ids(self.lineup, self.events)
        events = SportecData.find_spadl_event_types(events)

        selected_cols = [
            "period_id",
            "utc_timestamp",
            "object_id",
            "spadl_type",
            "coordinates_x",
            "coordinates_y",
            "success",
        ]
        input_events = events.loc[events["spadl_type"].notna(), selected_cols].copy()
        input_events.columns = selected_cols[:2] + ["player_id", "event_type", "start_x", "start_y", "success"]

        return input_events
