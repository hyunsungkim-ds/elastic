import os
import sys
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

from sync import config, schema, utils


class ReceiveDetector:
    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, fps: int = 25) -> None:
        schema.synced_event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        # Ensure unique indices
        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()

        pass_like = config.PASS_LIKE_OPEN + config.SET_PIECE  # + ["interception"]
        self.passes = self.events[self.events["spadl_type"].isin(pass_like) & self.events["frame_id"].notna()].copy()

        self.tracking = tracking
        self.players = self.tracking["player_id"].dropna().unique()

        self.fps = fps

        # Define an episode as a sequence of consecutive in-play frames
        time_cols = ["frame_id", "period_id", "timestamp", "utc_timestamp"]
        self.frames = self.tracking[time_cols].drop_duplicates().sort_values("frame_id").set_index("frame_id")
        self.frames["timestamp"] = self.frames["timestamp"].apply(utils.seconds_to_timestamp)
        self.frames["episode_id"] = 0
        n_prev_episodes = 0

        for i in self.events["period_id"].unique():
            period_frames: pd.DataFrame = self.frames.loc[self.frames["period_id"] == i].index.values
            episode_ids = (np.diff(period_frames, prepend=-5) >= 5).astype(int).cumsum() + n_prev_episodes
            self.frames.loc[self.frames["period_id"] == i, "episode_id"] = episode_ids
            n_prev_episodes = episode_ids.max()

    @staticmethod
    def _find_best_frame(features: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        dist_valleys = find_peaks(-features["closest_dist"], prominence=0.5)[0]
        cand_idxs = [0] + dist_valleys.tolist() + [len(features) - 1]

        accel_peaks = find_peaks(features["ball_accel"], prominence=10, distance=10)[0]
        for i in accel_peaks:
            if not set(range(i - 3, i + 4)) & set(cand_idxs):
                cand_idxs.append(i)

        cand_features = features.iloc[np.sort(np.unique(cand_idxs))].copy()

        for i in cand_features.index:
            cand_features.at[i, "closest_dist"] = features.loc[i - 3 : i + 3, "closest_dist"].min()
            cand_features.at[i, "next_player_dist"] = features.loc[i - 3 : i + 3, "next_player_dist"].min()
            cand_features.at[i, "ball_accel"] = features.loc[i - 3 : i + 3, "ball_accel"].max()
            cand_features.at[i, "ball_height"] = features.loc[i - 3 : i + 3, "ball_height"].min()

        cand_features = cand_features[(cand_features["closest_dist"] < 2) & (cand_features["ball_height"] < 3)]

        if len(cand_features.index) == 0:
            return np.nan, cand_features

        elif len(cand_features.index) == 1:
            return cand_features.index[0], cand_features

        else:
            for i, frame in enumerate(cand_features.index):
                prev_frame = cand_features.index[i - 1] if i > 0 else features.index[0]
                next_player_max_dist = features["next_player_dist"].loc[prev_frame:frame].max()
                next_player_last_dist = features["next_player_dist"].at[frame]
                cand_features.at[frame, "kick_dist"] = next_player_max_dist - next_player_last_dist

            cand_features["score"] = utils.score_frames_receive(cand_features)
            return cand_features["score"].idxmax(), cand_features

    def _detect_receive(self, event_idx: int, s: float = 10) -> Tuple[float, str, pd.DataFrame, pd.DataFrame]:
        pass_frame = self.events.at[event_idx, "frame_id"]

        episode_id = self.frames.at[pass_frame, "episode_id"]
        episode_last_frame = float(self.frames[self.frames["episode_id"] == episode_id].index[-1])
        max_frame = min(pass_frame + self.fps * s, episode_last_frame)

        event_type = self.events.at[event_idx, "spadl_type"]
        shot_types = ["shot", "shot_freekick", "shot_penalty"]
        pass_types = ["pass", "cross", "freekick_crossed", "freekick_short"] + config.SET_PIECE_OOP

        next_type = self.events.at[event_idx, "next_type"]
        next_event_frame = self.events.loc[event_idx + 1 :, "frame_id"].dropna().min()
        next_episode_id = self.frames.at[next_event_frame, "episode_id"] if not pd.isna(next_event_frame) else np.nan

        if not pd.isna(next_event_frame):
            max_frame = min(max_frame, next_event_frame)

        if next_type is None:
            # End of a period
            return episode_last_frame, None, None, None

        elif next_type in config.SET_PIECE_OOP:
            # End of an episode (i.e., the game is paused after the event)
            return episode_last_frame, "out", None, None

        elif event_type in shot_types and self.events.at[event_idx, "success"]:
            # Scoring a goal
            return episode_last_frame, "goal", None, None

        elif (
            next_episode_id == episode_id
            and next_type in config.INCOMING + ["shot_block", "keeper_punch"]
            and not pd.isna(next_event_frame)
        ):
            # The next event is already a receive event
            return next_event_frame, self.events.at[event_idx, "next_player_id"], None, None

        elif pass_frame == max_frame:
            # The pass and the next event occurs almost immediately
            return max_frame, self.events.at[event_idx, "next_player_id"], None, None

        else:
            window = self.tracking[(self.tracking["frame_id"] >= pass_frame) & (self.tracking["frame_id"] <= max_frame)]

            passer = self.events.at[event_idx, "player_id"]
            next_player = self.events.at[event_idx, "next_player_id"]

            players = [p for p in window["player_id"].unique() if p is not None]
            if event_type in pass_types:
                if self.events.at[event_idx, "success"]:
                    players = [p for p in players if p[:4] == passer[:4] and p != passer]  # Teammates are candidates
                else:
                    players = [p for p in players if p[:4] != passer[:4]]  # Opponents are candidates

            player_x = window[window["player_id"].isin(players)].pivot_table("x", "frame_id", "player_id", "first")
            player_y = window[window["player_id"].isin(players)].pivot_table("y", "frame_id", "player_id", "first")
            ball_window = window[window["ball"]]

            features = pd.DataFrame(index=player_x.index)
            features["ball_height"] = ball_window["z"].values
            features["ball_accel"] = ball_window["accel_v"].values

            next_player_window = window[window["player_id"] == next_player]
            if next_player_window.empty:
                return np.nan, None, None, None

            next_player_dist_x = next_player_window["x"].values - ball_window["x"].values
            next_player_dist_y = next_player_window["y"].values - ball_window["y"].values
            next_player_dists = np.sqrt(next_player_dist_x**2 + next_player_dist_y**2)
            features["next_player_dist"] = next_player_dists if next_player != passer else 0

            if self.events.at[event_idx, "spadl_type"] == "clearance":
                features["closest_dist"] = features["next_player_dist"]
                best_frame, cand_features = ReceiveDetector._find_best_frame(features)
                return float(best_frame), next_player, features, cand_features.index.tolist()

            if next_type in ["ball_touch", "shot_block"]:
                features["closest_dist"] = features["next_player_dist"]
                best_frame, cand_features = ReceiveDetector._find_best_frame(features)
                return float(best_frame), next_player, features, cand_features.index.tolist()

            else:
                player_dist_x = player_x[players].values - ball_window[["x"]].values
                player_dist_y = player_y[players].values - ball_window[["y"]].values
                player_dists = np.sqrt(player_dist_x**2 + player_dist_y**2)

                features["closest_id"] = np.array(players)[np.nanargmin(player_dists, axis=1)]
                features["closest_dist"] = np.nanmin(player_dists, axis=1)

                best_frame, cand_features = ReceiveDetector._find_best_frame(features)
                receiver = features.at[best_frame, "closest_id"] if best_frame == best_frame else None
                return float(best_frame), receiver, features, cand_features

    def run(self, s=10) -> None:
        self.events["receiver_id"] = None
        self.events["receive_frame_id"] = np.nan

        for pass_idx in tqdm(self.passes.index, desc="Detecting receiving events"):
            frame, receiver, _, _ = self._detect_receive(pass_idx, s)
            self.passes.at[pass_idx, "receiver_id"] = receiver
            self.passes.at[pass_idx, "receive_frame_id"] = frame
            self.events.at[pass_idx, "receiver_id"] = receiver
            self.events.at[pass_idx, "receive_frame_id"] = frame

    def plot_window_features(self, pass_idx: int, display_title: bool = True, save_path: str = None) -> pd.DataFrame:
        best_frame, receiver, features, cand_features = self._detect_receive(pass_idx)

        pass_type = self.events.at[pass_idx, "spadl_type"]
        passer = self.events.at[pass_idx, "player_id"]
        print(f"Current event: {pass_type} by {passer}")

        next_type = self.events.at[pass_idx, "next_type"]
        next_player = self.events.at[pass_idx, "next_player_id"]
        print(f"Next event: {next_type} by {next_player}")

        if not pd.isna(best_frame):
            matched_period = self.frames.at[best_frame, "period_id"]
            matched_time = self.frames.at[best_frame, "timestamp"]
            print(f"\nDetected receiver: {receiver}")
            print(f"Receiving frame: {best_frame}")
            print(f"Receiving time: P{matched_period}-{matched_time}")

        if isinstance(features, pd.DataFrame) and not features.empty:
            features["ball_accel"] = features["ball_accel"] / 5
            features_to_plot = ["closest_dist", "ball_accel", "next_player_dist"]

            plt.rcParams.update({"font.size": 15})
            plt.figure(figsize=(8, 4))
            plt.plot(features[features_to_plot], label=features_to_plot)

            ymax = 25
            plt.ylim(0, ymax)

            if isinstance(cand_features, pd.DataFrame):
                cand_frames = [t for t in cand_features.index if t != best_frame]
                plt.vlines(cand_frames, 0, ymax, color="black", linestyles="--", label="cand_frame")
                plt.vlines(best_frame, 0, ymax, color="red", linestyles="-", label="best_frame")

            elif not pd.isna(best_frame):
                plt.vlines(best_frame, 0, ymax, color="red", linestyles="-", label="best_frame")

            plt.legend(loc="upper right", fontsize=12)
            plt.grid(axis="y")
            plt.xticks(rotation=45)

            if display_title and not pd.isna(best_frame):
                plt.title(f"receive at frame {int(best_frame)}")

            if save_path is not None:
                plt.savefig(save_path, bbox_inches="tight")

            plt.show()

            return cand_features
