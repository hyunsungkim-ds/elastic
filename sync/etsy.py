import os
import sys
from typing import Callable, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sync import config, schema, utils


class ETSY:
    """Synchronize event and tracking data using the ETSY algorithm.

    Parameters
    ----------
    events : pd.DataFrame
        Event data to synchronize, according to schema etsy.schema.event_schema.
    tracking : pd.DataFrame
        Tracking data to synchronize, according to schema etsy.schema.tracking_schema.
    fps : int
        Recording frequency (frames per second) of the tracking data.
    kickoff_time : int
        Length of the window (in seconds) at the start of a playing period in which to search for the kickoff frame.
    """

    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, fps: int = 25):
        schema.etsy_event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        # Ensure unique index
        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()
        self.tracking = tracking
        self.fps = fps

        time_cols = ["frame_id", "period_id", "timestamp", "utc_timestamp"]
        self.frames = self.tracking[time_cols].drop_duplicates().sort_values("frame_id").set_index("frame_id")
        self.frames["timestamp"] = self.frames["timestamp"].apply(utils.seconds_to_timestamp)

        # Store synchronization results
        self.last_matched_frame = 0
        self.matched_frames = pd.Series(np.nan, index=self.events.index)
        self.scores = pd.Series(np.nan, index=self.events.index)

    def detect_kickoff(self, period: int) -> int:
        """Searches for the kickoff frame in a given playing period.

        Parameters
        ----------
        period: int
            The given playing period.

        Returns
        -------
            The detected kickoff frame.
        """
        kickoff_event = self.events[self.events["period_id"] == period].iloc[0]

        if kickoff_event["spadl_type"] != "pass":
            raise Exception("First event is not a pass!")

        frame = self.tracking.loc[self.tracking["period_id"] == period, "frame_id"].min()
        frames_to_check = np.arange(frame, frame + self.fps * config.TIME_KICKOFF)
        kickoff_player = kickoff_event["player_id"]

        inside_center_circle = self.tracking[
            (self.tracking["frame_id"] == frame)
            & (self.tracking["player_id"].str.startswith(kickoff_player.split("_")[0]))
            & (self.tracking["x"] >= config.PITCH_X / 2 - 5)
            & (self.tracking["x"] <= config.PITCH_X / 2 + 5)
            & (self.tracking["y"] >= config.PITCH_Y / 2 - 5)
            & (self.tracking["y"] <= config.PITCH_Y / 2 + 5)
        ]
        if len(inside_center_circle) > 1:
            print("Multiple players inside the center circle at kickoff!")
            raise ValueError

        ball_window: pd.DataFrame = self.tracking[
            (self.tracking["frame_id"].isin(frames_to_check))
            & (self.tracking["period_id"] == period)
            & self.tracking["ball"]
            & (self.tracking["x"] >= config.PITCH_X / 2 - 3)
            & (self.tracking["x"] <= config.PITCH_X / 2 + 3)
            & (self.tracking["y"] >= config.PITCH_Y / 2 - 3)
            & (self.tracking["y"] <= config.PITCH_Y / 2 + 3)
        ]
        ball_window = ball_window[(ball_window["frame_id"].diff() > 1).astype(int).cumsum() < 1].set_index("frame_id")

        if ball_window.empty:
            print("The tracking data begins after kickoff!")
            raise ValueError

        player_window: pd.DataFrame = self.tracking[
            (self.tracking["frame_id"].isin(frames_to_check))
            & (self.tracking["period_id"] == period)
            & (self.tracking["player_id"] == kickoff_player)
        ]
        player_window = player_window.set_index("frame_id").loc[ball_window.index]

        player_x = player_window["x"].values
        player_y = player_window["y"].values
        ball_x = ball_window["x"].values
        ball_y = ball_window["y"].values
        dists = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
        dist_idxs = np.where(dists < 2.0)[0]

        if len(dist_idxs) == 0:
            best_idx = np.argmin(dists)
        else:
            best_idx = ball_window["accel_s"].values[dist_idxs].argmax()

        return player_window.reset_index()["frame_id"].iloc[best_idx]

    def _window_of_frames(
        self, event: pd.Series, s: int = 5, min_frame: int = 0
    ) -> Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Identifies the qualifying window of frames around the event's timestamp.

        Parameters
        ----------
        event: pd.Series
            The record of the event to be matched.
        s: int
            Window length (in seconds).
        min_frame: int
            Minimum frame of windows (typically set as the timestamp of the previously detected event).
        max_frame: int
            Maximum frame of windows (typically set as the timestamp of the next detected event).

        Returns
        -------
        event_frame: int
            The closest frame to the recorded event timestamp.
        player_window: pd.DataFrame
            All frames of the acting player in the given window.
        ball_window: pd.DataFrame
            All frames containing the ball in the given window.
        """
        # Find the closest tracking frame and get candidate frames to search through
        event_frame = abs(self.frames["utc_timestamp"] - event["utc_timestamp"]).idxmin()
        cand_frames = np.arange(event_frame - self.fps * s + 1, event_frame + self.fps * s)

        # Select all player and ball frames within window range
        cand_frames = [t for t in cand_frames if t >= min_frame]
        window = self.tracking[self.tracking["frame_id"].isin(cand_frames)].copy()
        player_window: pd.DataFrame = window[window["player_id"] == event["player_id"]].set_index("frame_id")
        ball_window: pd.DataFrame = window[window["ball"]].set_index("frame_id")

        window_idxs = player_window.index.intersection(ball_window.index)
        player_window = player_window.loc[window_idxs].copy()
        ball_window = ball_window.loc[window_idxs].copy()

        return event_frame, player_window, ball_window

    @staticmethod
    def _mask_pass_like(features: pd.DataFrame) -> pd.DataFrame:
        dist_mask = features["player_ball_dist"] <= 2.5
        height_mask = features["ball_height"] <= 3
        accel_mask = features["ball_accel_s"] >= -1
        return features[dist_mask & height_mask & accel_mask].copy()

    @staticmethod
    def _mask_incoming(features: pd.DataFrame) -> pd.DataFrame:
        dist_mask = features["player_ball_dist"] <= 2
        height_mask = features["ball_height"] <= 3
        accel_mask = features["ball_accel_s"] <= 1
        return features[dist_mask & height_mask & accel_mask].copy()

    @staticmethod
    def _mask_bad_touch(features: pd.DataFrame) -> pd.DataFrame:
        dist_mask = features["player_ball_dist"] <= 3
        height_mask = features["ball_height"] <= 3
        return features[dist_mask & height_mask].copy()

    @staticmethod
    def _mask_fault_like(features: pd.DataFrame) -> pd.DataFrame:
        dist_mask = features["player_ball_dist"] <= 3
        height_mask = features["ball_height"] <= 4
        return features[dist_mask & height_mask].copy()

    @staticmethod
    def _find_masking_func(event_type: str) -> Tuple[float, Callable]:
        if event_type in config.PASS_LIKE_OPEN:
            s = config.TIME_PASS_LIKE_OPEN
            matching_func = ETSY._mask_pass_like
        elif event_type in config.SET_PIECE:
            s = config.TIME_SET_PIECE
            matching_func = ETSY._mask_pass_like
        elif event_type in config.INCOMING:
            s = config.TIME_INCOMING
            matching_func = ETSY._mask_incoming
        elif event_type in config.BAD_TOUCH:
            s = config.TIME_BAD_TOUCH
            matching_func = ETSY._mask_bad_touch
        elif event_type in config.FAULT_LIKE:
            s = config.TIME_FAULT_LIKE
            matching_func = ETSY._mask_fault_like
        else:
            s = 0
            matching_func = None
        return s, matching_func

    def _find_matching_frame(
        self,
        masking_func: Callable,
        event_idx: int,
        player_window: pd.DataFrame,
        ball_window: pd.DataFrame,
    ) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
        """Finds the matching frame of the given event within the given window.

        Parameters
        ----------
        matching_func: Callable
            One of the action-specific matching function, depending on the event's type.
        event_idx: int
            The index of the event to be matched.
        event_frame: int
            The closest frame to the recorded event timestamp.
        player_window: pd.DataFrame
            All frames of the acting player within a certain window.
        ball_window: pd.DataFrame
            All frames of the ball within the same window.

        Returns
        -------
        best_frame: int
            Index of the matching frame in the tracking dataframe.
        features: pd.DataFrame
            Features for each frame in the window that is used for matching.
        """
        player_x = player_window["x"].values
        player_y = player_window["y"].values
        ball_x = ball_window["x"].values
        ball_y = ball_window["y"].values
        event_x = self.events.at[event_idx, "start_x"]
        event_y = self.events.at[event_idx, "start_y"]

        features = pd.DataFrame(index=player_window.index)
        features["ball_accel_s"] = ball_window["accel_s"].values
        features["ball_height"] = ball_window["z"].values

        features["player_ball_dist"] = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
        features["player_event_dist"] = np.sqrt((player_x - event_x) ** 2 + (player_y - event_y) ** 2)
        features["ball_event_dist"] = np.sqrt((ball_x - event_x) ** 2 + (ball_y - event_y) ** 2)

        cand_features: pd.DataFrame = masking_func(features)

        if len(cand_features.index) == 0:
            return np.nan, features, None
        elif len(cand_features.index) == 1:
            return cand_features.index[0], features, None
        else:
            cand_features["score"] = utils.score_frames_etsy(cand_features)
            return cand_features["score"].idxmax(), features, cand_features

    def _sync_period_events(self, period: int) -> None:
        """Synchronizes the event and tracking data of a given playing period.

        Parameters
        ----------
        period: int
            The playing period of which to synchronize the event and tracking data.
        """
        period_events = self.events[self.events["period_id"] == period].copy()

        for i in tqdm(period_events.index[1:], desc=f"Syncing events in period {period}"):
            event_type = self.events.at[i, "spadl_type"]
            s, matching_func = ETSY._find_masking_func(event_type)
            _, player_window, ball_window = self._window_of_frames(self.events.loc[i], s, self.last_matched_frame)

            if len(player_window) > 0:
                best_frame = self._find_matching_frame(matching_func, i, player_window, ball_window)[0]
                if best_frame == best_frame:
                    self.matched_frames[i] = best_frame
                    self.last_matched_frame = best_frame

    def run(self) -> None:
        """
        Applies the ETSY algorithm on the instantiated class.
        """
        kickoff_idx = 0

        for period in self.events["period_id"].unique():
            period_events: pd.DataFrame = self.events[self.events["period_id"] == period]

            try:  # Kick-off detection for the current period
                best_frame = self.detect_kickoff(period=period)
                self.last_matched_frame = best_frame
                self.matched_frames.loc[kickoff_idx] = best_frame

            except ValueError:  # If there is no candidate frames for the kickoff, then find the second event
                kickoff_frame = self.frames[self.frames["period_id"] == period].index[0]
                self.last_matched_frame = kickoff_frame
                self.matched_frames.loc[kickoff_idx] = kickoff_frame

                kickoff_idx += 1
                kickoff_event = self.events.loc[kickoff_idx]
                _, player_window, ball_window = self._window_of_frames(kickoff_event, config.TIME_KICKOFF)
                best_frame = self._find_matching_frame(ETSY._mask_pass_like, kickoff_idx, player_window, ball_window)[0]

            # Adjust the time bias between events and tracking
            ts_offset = self.events.at[kickoff_idx, "utc_timestamp"] - self.frames.at[best_frame, "utc_timestamp"]
            self.events.loc[period_events.index, "utc_timestamp"] -= ts_offset
            kickoff_idx = len(self.events[self.events["period_id"] == period])

            # Event synchronization for the current period
            self._sync_period_events(period)

        self.events["frame_id"] = self.matched_frames
        self.events["synced_ts"] = self.events["frame_id"].map(self.frames["timestamp"].to_dict())

    def plot_window_features(self, event_idx: int, display_title: bool = True, save_path: str = None) -> pd.DataFrame:
        """
        Plots the feature time-series for a given event for validation.

        Parameters
        ----------
        event_idx: int
            The index of the event to be matched.

        Returns
        -------
        features: pd.DataFrame
            Features for each frame in the window that is used for matching.
        """
        event = self.events.loc[event_idx]
        event_type = event["spadl_type"]

        s, masking_func = ETSY._find_masking_func(event_type)
        print(f"Event {event_idx}: {event_type} by {event['player_id']}")

        prev_frames = self.matched_frames.loc[: event_idx - 1].values
        min_frame = np.nanmax([np.nanmax(prev_frames), 0])
        recorded_frame, p_window, b_window = self._window_of_frames(event, s, min_frame)

        best_frame, features, cand_features = self._find_matching_frame(masking_func, event_idx, p_window, b_window)

        if not pd.isna(best_frame):
            matched_period = self.frames.at[best_frame, "period_id"]
            matched_time = self.frames.at[best_frame, "timestamp"]
            print(f"Matched frame: {best_frame}")
            print(f"Matched time: P{matched_period}-{matched_time}")

        elif abs(self.frames.index.values - recorded_frame).min() < self.fps * config.TIME_SET_PIECE:
            recorded_frame = self.frames.index[abs(self.frames.index.values - recorded_frame).argmin()]
            period = self.frames.at[recorded_frame, "period_id"]
            recorded_ts = self.frames.at[recorded_frame, "timestamp"]
            print(f"Recorded frame: {recorded_frame}")
            print(f"Closest time: P{period}-{recorded_ts}")

        else:
            print(f"Recorded frame: {recorded_frame}")
            print("Out-of-play at the recorded frame.")

        if features.empty:
            return

        else:
            features["ball_accel"] = features["ball_accel_s"] / 5

            plt.close("all")
            plt.rcParams.update({"font.size": 15})
            plt.figure(figsize=(10, 5))

            # colors = {
            #     "ball_accel": "tab:orange",
            #     "player_ball_dist": "tab:blue",
            #     "player_event_dist": "tab:green",
            #     "ball_event_dist": "tab:purple",
            # }
            # for feat, color in colors.items():
            #     plt.plot(features[feat], label=feat, c=color)

            plt.plot(features["player_ball_dist"], label="player-ball dist.", color="tab:blue")
            plt.plot(features["player_event_dist"], label="player-event dist.", color="tab:green")
            plt.plot(features["ball_event_dist"], label="event-ball dist.", color="tab:purple")

            ymax = 25
            plt.xticks()
            plt.ylim(0, ymax)
            # plt.vlines(windows[0], 0, ymax, color="k", linestyles="-", label="annot_frame")

            # if isinstance(cand_features, pd.DataFrame):
            #     cand_frames = [t for t in cand_features.index if t != best_frame]
            #     plt.vlines(cand_frames, 0, ymax, color="black", linestyles="--", label="cand_frame")
            #     plt.vlines(best_frame, 0, ymax, color="red", linestyles="-", label="best_frame")

            # elif not pd.isna(best_frame):
            #     plt.vlines(best_frame, 0, ymax, color="red", linestyles="-", label="best_frame")

            plt.legend(loc="upper right", fontsize=15)
            plt.grid(axis="y")

            if display_title:
                plt.title(f"{event_type} at frame {best_frame}")

            if save_path is not None:
                plt.savefig(save_path, bbox_inches="tight")

            plt.show()

        return cand_features
