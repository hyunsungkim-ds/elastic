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
from tools.preprocessor import Preprocessor


class SportecProcessor:
    def __init__(self):
        self.meta_dir = "data/sportec/metadata"
        self.event_dir = "data/sportec/event"
        self.tracking_dir = "data/sportec/tracking"

        self.pitch_dims = MetricPitchDimensions(standardized=True, x_dim=Dimension(0, 105), y_dim=Dimension(0, 68))
        self.position_mapping: Dict[str, str] = {
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

    def get_meta_path(self, match_id: str) -> str:
        meta_file = [f for f in os.listdir(self.meta_dir) if "matchinformation" in f and match_id in f][0]
        return f"{self.meta_dir}/{meta_file}"

    def get_event_path(self, match_id: str) -> str:
        event_file = [f for f in os.listdir(self.event_dir) if "events" in f and match_id in f][0]
        return f"{self.event_dir}/{event_file}"

    def get_tracking_path(self, match_id: str) -> str:
        tracking_file = [f for f in os.listdir(self.tracking_dir) if "positions" in f and match_id in f][0]
        return f"{self.tracking_dir}/{tracking_file}"

    def load_player_metadata(self, match_id: str) -> pd.DataFrame:
        meta_path = self.get_meta_path(match_id)

        tree = ET.parse(meta_path)
        root = tree.getroot()
        player_list = []

        for team in root.findall(".//Team"):
            team_id = team.attrib.get("TeamId")
            team_name = team.attrib.get("TeamName")
            home_away = "away" if team.attrib.get("Role") == "guest" else "home"

            players = team.find("Players")
            if players is not None:
                for player in players.findall("Player"):
                    uniform_number = int(player.attrib.get("ShirtNumber"))
                    player_list.append(
                        {
                            "team_id": team_id,
                            "team_name": team_name,
                            "home_away": home_away,
                            "player_id": player.attrib.get("PersonId"),
                            "uniform_number": uniform_number,
                            "object_id": f"{home_away}_{uniform_number}",
                            "player_name": player.attrib.get("Shortname"),
                            "starting": player.attrib.get("Starting") == "true",
                            "playing_position": self.position_mapping[player.attrib.get("PlayingPosition")],
                            "captain": player.attrib.get("TeamLeader") == "true",
                        }
                    )

        return pd.DataFrame(player_list).sort_values(["home_away", "uniform_number"], ignore_index=True)

    def parse_event_data(event_path: str) -> pd.DataFrame:
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
                success = play.find("SuccessfulShot")

                if play.find("ShotWide") is not None:
                    result = "OffTarget"
                elif play.find("SavedShot") is not None:
                    result = "Saved"
                elif play.find("BlockedShot") is not None:
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

            elif child.tag == "Caution":
                event_type = "Card"
                if child.attrib.get("CardColor") == "yellow":
                    card_type = "Yellow"
                elif child.attrib.get("CardColor") == "red":
                    card_type = "Red"

            elif child.tag == "OtherBallAction":
                if child.attrib.get("DefensiveClearance") == "true":
                    event_type = "Clearance"
                else:
                    event_type = "OtherBallAction"

            event_rows.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "period_id": period_id,
                    "timestamp": timestamp,
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

        event_df = pd.DataFrame(event_rows)
        event_df["datetime"] = pd.to_datetime(event_df["timestamp"]).dt.tz_localize(None)
        event_df["timestamp"] = None
        event_df.sort_values("datetime", ignore_index=True, inplace=True)

        period_ids = [id for id in event_df["period_id"].unique() if id > 0]
        for period_id in period_ids:
            period_events = event_df[event_df["period_id"] == period_id].copy()
            start_idx = period_events[period_events["set_piece_type"] == "KickOff"].index[0]
            end_idx = period_events[period_events["event_type"] == "FinalWhistle"].index[-1]

            start_dt = event_df.at[start_idx, "datetime"]
            event_df.loc[start_idx:end_idx, "period_id"] = period_id
            event_df.loc[start_idx:end_idx, "timestamp"] = event_df.loc[start_idx:end_idx, "datetime"] - start_dt

        return event_df.drop("datetime", axis=1)

    def load_event_data(
        self,
        match_id: str,
        player_df: pd.DataFrame,
        filter_valid_actions=True,
        find_object_ids=True,
    ) -> pd.DataFrame:
        # meta_path = self.get_meta_path(match_id)
        event_path = self.get_event_path(match_id)

        print(f"Loading the event data for Match {match_id}...")
        # event_ds = sportec.load_event(event_data=event_path, meta_data=meta_path)
        # event_ds = event_ds.transform(to_orientation=Orientation.HOME_AWAY, to_pitch_dimensions=self.pitch_dims)
        # event_df = event_ds.to_df().sort_values(["period_id", "timestamp"], ignore_index=True)
        event_df = SportecProcessor.parse_event_data(event_path)

        events_list = []

        for period_id in event_df["period_id"].unique():
            period_events = event_df[event_df["period_id"] == period_id]
            kickoffs = period_events[period_events["set_piece_type"] == "KICK_OFF"]
            if not kickoffs.empty:
                period_events = period_events.loc[kickoffs.index[0] :]
            events_list.append(period_events)

        event_df = pd.concat(events_list, ignore_index=True)

        if filter_valid_actions:
            drop_types = [
                "Delete",
                # "TacklingGame",
                # "SitterPrevented",
                # "SpectacularPlay",
                # "PenaltyNotAwarded",
                "VideoAssistantAction",
                "RefereeBall",
                "Card",
                "Substitution",
            ]
            event_df = event_df[~event_df["event_type"].isin(drop_types)].copy()

            # ball_out_mask = (event_df["event_type"].shift(-1) == "BALL_OUT") & (event_df["result"].isna())
            # event_df.loc[ball_out_mask, "result"] = "OUT"
            # event_df.loc[event_df["event_type"].shift(-1) == "Offside", "result"] = "OFFSIDE"
            # event_df = event_df[~event_df["event_type"].isin(["BALL_OUT", "Offside"])]
            # event_df = event_df.reset_index(drop=True).copy()

        if find_object_ids:
            player_mapping = player_df.set_index("player_id")["object_id"].to_dict()
            event_df["object_id"] = event_df["player_id"].map(player_mapping)
            event_df["receiver_id"] = event_df["receiver_player_id"].map(player_mapping)
            event_df.loc[event_df["event_type"] == "VideoAssistantAction", "object_id"] = "referee"
            event_df.loc[event_df["event_type"] == "RefereeBall", "object_id"] = "referee"
            event_df.loc[event_df["event_type"] == "PenaltyNotAwarded", "object_id"] = "referee"

        return event_df

    def load_tracking_data(self, match_id: str, player_df: pd.DataFrame) -> Tuple[TrackingDataset, pd.DataFrame]:
        meta_path = self.get_meta_path(match_id)
        tracking_path = self.get_tracking_path(match_id)

        print(f"Loading the tracking data for Match {match_id}")
        tracking_ds = sportec.load_tracking(raw_data=tracking_path, meta_data=meta_path, only_alive=False)

        print("Transforming the tracking data coordinates...")
        tracking_ds = tracking_ds.transform(to_orientation=Orientation.HOME_AWAY, to_pitch_dimensions=self.pitch_dims)

        tracking_df: pd.DataFrame = tracking_ds.to_df()
        player_mapping = player_df.set_index("player_id")["object_id"].to_dict()
        column_mapping = {f"{k}_{t}": f"{v}_{t}" for k, v in player_mapping.items() for t in ["x", "y", "d", "s"]}
        tracking_df = tracking_df.rename(columns=column_mapping)

        player_x_cols = [c for c in tracking_df.columns if fnmatch(c, "home_*_x") or fnmatch(c, "away_*_x")]
        tracking_df = tracking_df.dropna(subset=player_x_cols, how="all").copy()

        return tracking_ds, tracking_df

    @staticmethod
    def find_spadl_event_types(event_df: pd.DataFrame) -> pd.DataFrame:
        event_df["spadl_type"] = None

        # NOTE: In the current version of Sportec Open Data, both passes and crosses are annotated as PASS
        pass_mask = event_df["event_type"] == "PASS"
        event_df.loc[pass_mask, "spadl_type"] = "pass"
        event_df.loc[pass_mask & (event_df["set_piece_type"] == "THROW_IN"), "spadl_type"] = "throw_in"
        event_df.loc[pass_mask & (event_df["set_piece_type"] == "GOAL_KICK"), "spadl_type"] = "goalkick"
        event_df.loc[pass_mask & (event_df["set_piece_type"] == "CORNER_KICK"), "spadl_type"] = "corner_short"
        event_df.loc[pass_mask & (event_df["set_piece_type"] == "FREE_KICK"), "spadl_type"] = "freekick_short"

        cross_mask = event_df["event_type"] == "CROSS"
        event_df.loc[cross_mask, "spadl_type"] = "cross"
        event_df.loc[cross_mask & (event_df["set_piece_type"] == "CORNER_KICK"), "spadl_type"] = "corner_crossed"
        event_df.loc[cross_mask & (event_df["set_piece_type"] == "FREE_KICK"), "spadl_type"] = "freekick_crossed"

        shot_mask = event_df["event_type"] == "SHOT"
        event_df.loc[shot_mask, "spadl_type"] = "shot"
        event_df.loc[shot_mask & (event_df["set_piece_type"] == "FREE_KICK"), "spadl_type"] = "shot_freekick"
        event_df.loc[shot_mask & (event_df["set_piece_type"] == "PENALTY"), "spadl_type"] = "shot_penalty"

        event_df.loc[event_df["event_type"] == "RECOVERY", "spadl_type"] = "ball_recovery"

        # NOTE: In the current version of Sportec Open Data,
        # GENERIC:OtherBallAction is a mixture of multiple defensive action categories,
        # including interception, tackle, clearance, and so on.
        event_df.loc[event_df["event_type"] == "GENERIC:OtherBallAction", "spadl_type"] = "bad_touch"
        event_df.loc[event_df["event_type"] == "FOUL_COMMITTED", "spadl_type"] = "foul"

        return event_df
