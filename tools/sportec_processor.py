import os
import xml.etree.ElementTree as ET
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
from kloppy import sportec
from kloppy.domain import Dimension, MetricPitchDimensions, Orientation
from scipy.signal import savgol_filter

from sync import config
from tools.preprocessor import Preprocessor


class SportecProcessor:
    def __init__(self, match_id: str):
        meta_file = [f for f in os.listdir("data/sportec/metadata") if match_id in f][0]
        event_file = [f for f in os.listdir("data/sportec/event") if match_id in f][0]
        tracking_file = [f for f in os.listdir("data/sportec/tracking") if match_id in f][0]

        self.meta_path = f"data/sportec/metadata/{meta_file}"
        self.event_path = f"data/sportec/event/{event_file}"
        self.tracking_path = f"data/sportec/tracking/{tracking_file}"

        self.players: pd.DataFrame = self.load_player_metadata()
        self.pitch_dims = MetricPitchDimensions(standardized=True, x_dim=Dimension(0, 105), y_dim=Dimension(0, 68))

        self.events = None
        self.tracking = None
        self.fps = None

    def load_player_metadata(self) -> pd.DataFrame:
        tree = ET.parse(self.meta_path)
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
                            "playing_position": player.attrib.get("PlayingPosition"),
                            "captain": player.attrib.get("TeamLeader") == "true",
                        }
                    )

        return pd.DataFrame(player_list).sort_values(["home_away", "uniform_number"], ignore_index=True)

    def load_event_data(self) -> pd.DataFrame:
        print("Loading the event data...")
        event_ds = sportec.load_event(event_data=self.event_path, meta_data=self.meta_path)
        event_ds = event_ds.transform(to_orientation=Orientation.HOME_AWAY, to_pitch_dimensions=self.pitch_dims)
        return event_ds.to_df().sort_values(["period_id", "timestamp"], ignore_index=True)

    def load_tracking_data(self) -> Tuple[pd.DataFrame, float]:
        print("Loading the tracking data...")
        tracking_ds = sportec.load_tracking(raw_data=self.tracking_path, meta_data=self.meta_path, only_alive=False)

        print("Transforming the tracking data coordinates...")
        tracking_ds = tracking_ds.transform(to_orientation=Orientation.HOME_AWAY, to_pitch_dimensions=self.pitch_dims)

        tracking = tracking_ds.to_df()
        player_dict = self.players.set_index("player_id")["object_id"].to_dict()
        col_dict = {f"{k}_{t}": f"{v}_{t}" for k, v in player_dict.items() for t in ["x", "y", "d", "s"]}
        tracking = tracking.rename(columns=col_dict).copy()

        return tracking, tracking_ds.frame_rate
