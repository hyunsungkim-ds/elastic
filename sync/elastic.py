import os
import sys
from typing import Callable, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm

from sync import config, schema, scoring
from sync.reception import ReceptionDetector


class ELASTIC:
    """Synchronize event and tracking data using Event-Location-AgnoSTIC Synchronizer (ELASTIC).

    Parameters
    ----------
    events: pd.DataFrame
        Event data to synchronize, according to schema sync.schema.event_schema.
    tracking: pd.DataFrame
        Tracking data to synchronize, according to schema sync.schema.tracking_schema.
    fps: int
        Recording frequency (frames per second) of the tracking data.
    """

    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, fps: int = 25) -> None:
        schema.event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        # Ensure unique indices
        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()
        self.tracking = tracking
        self.fps = fps

        # Define an episode as a sequence of consecutive in-play frames
        time_cols = ["frame", "period_id", "timestamp", "utc_timestamp"]
        self.frames = self.tracking[time_cols].drop_duplicates().sort_values("frame").set_index("frame")
        self.frames["timestamp"] = self.frames["timestamp"].apply(ELASTIC._format_timestamp)
        self.frames["episode_id"] = 0
        n_prev_episodes = 0

        for i in self.events["period_id"].unique():
            period_frames = self.frames.loc[self.frames["period_id"] == i].index.values
            episode_ids = (np.diff(period_frames, prepend=-5) >= 5).astype(int).cumsum() + n_prev_episodes
            self.frames.loc[self.frames["period_id"] == i, "episode_id"] = episode_ids
            n_prev_episodes = episode_ids.max()

        # Store synchronization results
        self.last_matched_frame = 0
        self.matched_frames = pd.Series(np.nan, index=self.events.index)
        self.reception_det = None

    @staticmethod
    def _format_timestamp(total_seconds: float) -> str:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{int(seconds):02d}{f'{seconds % 1:.2f}'[1:]}"

    def detect_kickoff(self, period: int, buffer_seconds=5) -> int:
        """Searches for the kickoff frame in a given playing period.

        Parameters
        ----------
        period: int
            The given playing period.
        kickoff_buffer: int
            Length of the search window (in seconds) for the kickoff frame of a playing period.

        Returns
        -------
            The detected kickoff frame.
        """
        kickoff_event = self.events[self.events["period_id"] == period].iloc[0]

        if kickoff_event["spadl_type"] != "pass":
            raise Exception("First event is not a pass!")

        frame = self.tracking.loc[self.tracking["period_id"] == period, "frame"].min()
        frames_to_check = np.arange(frame, frame + self.fps * buffer_seconds)
        kickoff_player = kickoff_event["player_id"]

        inside_center_circle = self.tracking[
            (self.tracking["frame"] == frame)
            & (self.tracking["player_id"].str.startswith(kickoff_player.split("_")[0]))
            & (self.tracking["x"] >= config.FIELD_LENGTH / 2 - 5)
            & (self.tracking["x"] <= config.FIELD_LENGTH / 2 + 5)
            & (self.tracking["y"] >= config.FIELD_WIDTH / 2 - 5)
            & (self.tracking["y"] <= config.FIELD_WIDTH / 2 + 5)
        ]
        if len(inside_center_circle) > 1:
            print("Multiple players inside the center circle at kickoff!")
            raise ValueError

        ball_window: pd.DataFrame = self.tracking[
            (self.tracking["frame"].isin(frames_to_check))
            & (self.tracking["period_id"] == period)
            & self.tracking["ball"]
            & (self.tracking["x"] >= config.FIELD_LENGTH / 2 - 3)
            & (self.tracking["x"] <= config.FIELD_LENGTH / 2 + 3)
            & (self.tracking["y"] >= config.FIELD_WIDTH / 2 - 3)
            & (self.tracking["y"] <= config.FIELD_WIDTH / 2 + 3)
        ]
        ball_window = ball_window[(ball_window["frame"].diff() > 1).astype(int).cumsum() < 1].set_index("frame")

        if ball_window.empty:
            print("The tracking data begins after kickoff!")
            raise ValueError

        player_window: pd.DataFrame = self.tracking[
            (self.tracking["frame"].isin(frames_to_check))
            & (self.tracking["period_id"] == period)
            & (self.tracking["player_id"] == kickoff_player)
        ]
        player_window = player_window.set_index("frame").loc[ball_window.index]

        player_x = player_window["x"].values
        player_y = player_window["y"].values
        ball_x = ball_window["x"].values
        ball_y = ball_window["y"].values
        dists = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
        dist_idxs = np.where(dists < 2.0)[0]

        if len(dist_idxs) == 0:
            best_idx = np.argmin(dists)
        else:
            best_idx = ball_window["accel"].values[dist_idxs].argmax()

        return player_window.reset_index()["frame"].iloc[best_idx]

    def _window_of_frames(
        self, event: pd.Series, s: int, duel_won=False
    ) -> Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Identifies the qualifying window of frames around the event's timestamp.

        Parameters
        ----------
        event: pd.Series
            The record of the event to be matched.
        s: int
            Window length (in seconds).

        Returns
        -------
        event_frame: int
            The closest frame to the recorded event timestamp.
        player_window: pd.DataFrame
            All frames of the acting player in the given window.
        ball_window: pd.DataFrame
            All frames containing the ball in the given window.
        """
        if event["spadl_type"] in config.SET_PIECE:
            frame_diffs = abs(self.frames["utc_timestamp"] - event["utc_timestamp"])

            if frame_diffs.min().total_seconds() > s:
                event_frame = None
                cand_frames = []
            else:
                event_frame = frame_diffs.idxmin()
                episode = self.frames.at[event_frame, "episode_id"]
                episode_frames = self.frames[self.frames["episode_id"] == episode].index

                if event_frame - episode_frames[0] < self.fps * s:
                    cand_frames = self.frames[self.frames["episode_id"] == episode].index[: self.fps]
                elif event["spadl_type"] == "throw_in":  # Fast throw-ins can happend in the middle of an episode.
                    cand_frames = np.arange(event_frame - self.fps * s + 1, event_frame + self.fps * s)
                else:  # Set-pieces except for throw-ins should start an episode. (Otherwise there must be an error.)
                    cand_frames = []

        elif event["spadl_type"] == "foul":
            prev_event_ts = self.events.at[event.name - 1, "utc_timestamp"]
            frame_diffs = abs(self.frames["utc_timestamp"] - prev_event_ts)

            if frame_diffs.min().total_seconds() > s:
                event_frame = None
                cand_frames = []
            else:
                prev_event_frame = frame_diffs.idxmin()
                episode = self.frames.at[prev_event_frame, "episode_id"]
                cand_frames = self.frames[self.frames["episode_id"] == episode].index[-self.fps :]
                event_frame = cand_frames[-1]

        else:
            # Find the closest tracking frame and get candidate frames to search through
            event_frame = abs(self.frames["utc_timestamp"] - event["utc_timestamp"]).idxmin()
            cand_frames = np.arange(event_frame - self.fps * s + 1, event_frame + self.fps * s)

        # Select all player and ball frames within window range
        window = self.tracking[self.tracking["frame"].isin(cand_frames)].copy()
        player_window: pd.DataFrame = window[window["player_id"] == event["player_id"]].set_index("frame")
        ball_window: pd.DataFrame = window[window["ball"]].set_index("frame")

        window_idxs = player_window.index.intersection(ball_window.index)
        player_window = player_window.loc[window_idxs].copy()
        ball_window = ball_window.loc[window_idxs].copy()

        prev_player_id = self.events.at[event.name - 1, "player_id"]
        if duel_won and event["player_id"].split("_")[0] != prev_player_id.split("_")[0]:
            oppo_window: pd.DataFrame = window[window["player_id"] == prev_player_id].set_index("frame")
            oppo_window = oppo_window.loc[window_idxs].copy()
        else:
            oppo_window = None

        return event_frame, player_window, ball_window, oppo_window

    @staticmethod
    def _detect_pass_like(features: pd.DataFrame, *args) -> Tuple[float, pd.DataFrame]:
        dist_valleys = find_peaks(-features["player_dist"], prominence=1)[0]
        candidates = dist_valleys.tolist() + [0]

        height_valleys = find_peaks(-features["ball_height"], prominence=0.5)[0]
        for i in height_valleys:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        accel_peaks = find_peaks(features["ball_accel"], prominence=10, distance=10)[0]
        for i in accel_peaks:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        candidates = np.sort(np.unique(candidates))
        cand_features = features.iloc[candidates].copy()

        for i in cand_features.index:
            cand_features.at[i, "player_dist"] = features.loc[i - 3 : i + 3, "player_dist"].min()
            cand_features.at[i, "ball_height"] = features.loc[i - 3 : i + 3, "ball_height"].min()
            cand_features.at[i, "ball_accel"] = features.loc[i - 3 : i + 3, "ball_accel"].max()

        cand_features = cand_features[(cand_features["player_dist"] < 3) & (cand_features["ball_height"] < 3.5)]

        if len(cand_features.index) == 0:
            return np.nan, features, None

        elif len(cand_features.index) == 1:
            return cand_features.index[0], features, None

        else:
            for i, frame in enumerate(cand_features.index):
                next_frame = cand_features.index[i + 1] if i < len(cand_features) - 1 else features.index[-1]
                cand_features.at[frame, "kick_dist"] = features["player_dist"].loc[frame:next_frame].max()

            cand_features["score"] = scoring.score_frames_elastic(cand_features)
            return cand_features["score"].idxmax(), features, cand_features.index

    @staticmethod
    def _detect_incoming(features: pd.DataFrame, savgol_wlen=9, fps=25) -> Tuple[float, pd.DataFrame, List[int]]:
        if len(features) > savgol_wlen:
            features["player_dist"] = savgol_filter(features["player_dist"], window_length=savgol_wlen, polyorder=2)

        features["rel_speed"] = features["player_dist"].diff().shift(-1).ffill() * fps
        features["rel_accel"] = abs(features["rel_speed"].diff().fillna(0) * fps)
        if len(features) > savgol_wlen:
            features["rel_accel"] = savgol_filter(features["rel_accel"], window_length=9, polyorder=2)

        dist_valleys = find_peaks(-features["player_dist"], prominence=1)[0]
        candidates = dist_valleys.tolist() + [0]

        height_valleys = find_peaks(-features["ball_height"], prominence=0.5)[0]
        for i in height_valleys:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        accel_peaks = find_peaks(features["rel_accel"], prominence=10, distance=5)[0]
        for i in accel_peaks:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        candidates = np.sort(np.unique(candidates))
        cand_features = features.iloc[candidates].copy()

        for i in cand_features.index:
            cand_features.at[i, "player_dist"] = features.loc[i - 3 : i + 3, "player_dist"].min()
            cand_features.at[i, "ball_height"] = features.loc[i - 3 : i + 3, "ball_height"].min()
            cand_features.at[i, "rel_speed"] = features.loc[i - 3 : i + 3, "rel_speed"].max()
            cand_features.at[i, "ball_accel"] = features.loc[i - 3 : i + 3, "rel_accel"].max()

        cand_features = cand_features[
            (cand_features["player_dist"] < 3)
            & (cand_features["ball_height"] < 3.5)
            & (cand_features["rel_speed"] > -1)
        ]

        if len(cand_features.index) == 0:
            return np.nan, features, None

        elif len(cand_features.index) == 1:
            return cand_features.index[0], features, None

        else:
            for i, frame in enumerate(cand_features.index):
                prev_frame = cand_features.index[i - 1] if i > 0 else features.index[0]
                cand_features.at[frame, "kick_dist"] = features["player_dist"].loc[prev_frame:frame].max()

            cand_features["score"] = scoring.score_frames_elastic(cand_features)
            return cand_features["score"].idxmax(), features, cand_features.index.tolist()

    @staticmethod
    def _detect_dispossessed(features: pd.DataFrame, *args) -> Tuple[float, pd.DataFrame, List[int]]:
        dist_valleys = find_peaks(-features["player_dist"], prominence=1)[0]
        candidates = dist_valleys.tolist() + [0]

        height_valleys = find_peaks(-features["ball_height"], prominence=0.5)[0]
        for i in height_valleys:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        accel_peaks = find_peaks(features["ball_accel"], prominence=10)[0]
        for i in accel_peaks:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        candidates = np.sort(np.unique(candidates))
        cand_features = features.iloc[candidates].copy()

        for i in cand_features.index:
            cand_features.at[i, "player_dist"] = features.loc[i - 3 : i + 3, "player_dist"].min()
            cand_features.at[i, "ball_height"] = features.loc[i - 3 : i + 3, "ball_height"].min()
            cand_features.at[i, "ball_accel"] = features.loc[i - 3 : i + 3, "ball_accel"].max()

        cand_features = cand_features[(cand_features["player_dist"] < 1.5) & (cand_features["ball_height"] < 3.5)]

        if len(cand_features.index) == 0:
            return np.nan, features, None

        elif len(cand_features.index) == 1:
            return cand_features.index[0], features, None

        else:
            for i, frame in enumerate(cand_features.index):
                next_frame = cand_features.index[i + 1] if i < len(cand_features) - 1 else features.index[-1]
                cand_features.at[frame, "kick_dist"] = features["player_dist"].loc[frame:next_frame].max()

            cand_features["score"] = scoring.score_frames_dispossessed(cand_features)
            return cand_features["score"].idxmax(), features, cand_features.index.tolist()

    @staticmethod
    def _detect_tackle(features: pd.DataFrame, savgol_wlen=9, fps=25) -> Tuple[float, pd.DataFrame, List[int]]:
        if len(features) > savgol_wlen:
            features["player_dist"] = savgol_filter(features["player_dist"], window_length=savgol_wlen, polyorder=2)

        features["rel_speed"] = features["player_dist"].diff().shift(-1).ffill() * fps
        features["rel_accel"] = abs(features["rel_speed"].diff().fillna(0) * fps)
        if len(features) > savgol_wlen:
            features["rel_accel"] = savgol_filter(features["rel_accel"], window_length=9, polyorder=2)

        player_dist_valleys = find_peaks(-features["player_dist"], prominence=1)[0] + 1
        candidates = player_dist_valleys.tolist() + [0]

        oppo_dist_valleys = find_peaks(-features["oppo_dist"], prominence=1)[0] + 1
        for i in oppo_dist_valleys:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        height_valleys = find_peaks(-features["ball_height"], prominence=0.5)[0]
        for i in height_valleys:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        accel_peaks = find_peaks(features["rel_accel"], prominence=10, distance=5)[0]
        for i in accel_peaks:
            if not set(range(i - 3, i + 4)) & set(candidates):
                candidates.append(i)

        candidates = np.sort(np.unique(candidates))
        cand_features = features.iloc[candidates].copy()

        for i in cand_features.index:
            cand_features.at[i, "player_dist"] = features.loc[i - 3 : i + 3, "player_dist"].min()
            cand_features.at[i, "oppo_dist"] = features.loc[i - 3 : i + 3, "oppo_dist"].min()
            cand_features.at[i, "ball_height"] = features.loc[i - 3 : i + 3, "ball_height"].min()
            cand_features.at[i, "rel_speed"] = features.loc[i - 3 : i + 3, "rel_speed"].max()
            cand_features.at[i, "ball_accel"] = features.loc[i - 3 : i + 3, "rel_accel"].max()

        cand_features = cand_features[
            (cand_features["player_dist"] < 3)
            & (cand_features["oppo_dist"] < 3)
            & (cand_features["ball_height"] < 3.5)
            & (cand_features["rel_speed"] > -1)
        ]

        if len(cand_features.index) == 0:
            return np.nan, features, None

        elif len(cand_features.index) == 1:
            return cand_features.index[0], features, None

        else:
            cand_features["score"] = scoring.score_frames_tackle(cand_features)
            # display(cand_features)
            return cand_features["score"].idxmax(), features, cand_features.index.tolist()

    @staticmethod
    def _detect_setpiece(features: pd.DataFrame, *args) -> Tuple[float, pd.DataFrame, None]:
        cand_features = features[features["player_dist"] < 2]
        best_frame = cand_features.index[0] if not cand_features.empty else np.nan
        return best_frame, features, None

    @staticmethod
    def _detect_foul(features: pd.DataFrame, *args) -> Tuple[float, pd.DataFrame, None]:
        if features.empty:
            return np.nan, features, None
        else:
            frames = features.index.to_numpy()
            gap_indices = np.where(np.diff(frames) > 1)[0]
            pause_frame = frames[gap_indices[0]] if len(gap_indices) > 0 else frames[-1]
            return pause_frame, features, None

    @staticmethod
    def _find_matching_func(event_type: str) -> Tuple[float, Callable]:
        if event_type in config.PASS_LIKE_OPEN + ["bad_touch"]:
            s = config.TIME_PASS_LIKE_OPEN
            matching_func = ELASTIC._detect_pass_like
        elif event_type in config.INCOMING + ["take_on"]:
            s = config.TIME_INCOMING
            matching_func = ELASTIC._detect_incoming
        elif event_type in config.SET_PIECE:
            s = config.TIME_SET_PIECE
            matching_func = ELASTIC._detect_setpiece
        elif event_type == "dispossessed":
            s = config.TIME_PASS_LIKE_OPEN
            matching_func = ELASTIC._detect_dispossessed
        elif event_type == "tackle":
            s = config.TIME_INCOMING
            matching_func = ELASTIC._detect_tackle
        elif event_type == "foul":
            s = config.TIME_FAULT_LIKE
            matching_func = ELASTIC._detect_foul
        else:
            s = 0
            matching_func = None
        return s, matching_func

    def _find_matching_frame(
        self,
        matching_func: Callable,
        event_idx: int,
        event_frame: int,
        player_window: pd.DataFrame,
        ball_window: pd.DataFrame,
        oppo_window: pd.DataFrame = None,
    ) -> Tuple[float, pd.DataFrame]:
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
        oppo_window (optional): pd.DataFrame
            All frames of the dueling opponent within the same window.

        Returns
        -------
        best_frame: int
            Index of the matching frame in the tracking dataframe.
        features: pd.DataFrame
            Features for each frame in the window that is used for matching.
        """
        ball_x = ball_window["x"].values
        ball_y = ball_window["y"].values
        player_x = player_window["x"].values
        player_y = player_window["y"].values
        # event_x = self.events.at[event_idx, "start_x"]
        # event_y = self.events.at[event_idx, "start_y"]

        features = pd.DataFrame(index=player_window.index)
        features["frame_delay"] = (features.index.values - event_frame).clip(0)
        features["ball_accel"] = ball_window["accel"].values
        features["ball_height"] = ball_window["z"].values
        features["player_dist"] = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
        # features["event_dist"] = np.sqrt((event_x - ball_x) ** 2 + (event_y - ball_y) ** 2)

        if oppo_window is not None:
            oppo_x = oppo_window["x"].values
            oppo_y = oppo_window["y"].values
            features["oppo_dist"] = np.sqrt((oppo_x - ball_x) ** 2 + (oppo_y - ball_y) ** 2)

        return matching_func(features)

    def _sync_period_events(self, period: int) -> None:
        """Synchronizes the event and tracking data of a given playing period.

        Parameters
        ----------
        period: int
            The playing period of which to synchronize the event and tracking data.
        """
        idxs = self.events[self.events["period_id"] == period].index

        for i in tqdm(idxs[1:], desc=f"Syncing major events in period {period}"):
            event_type = self.events.at[i, "spadl_type"]
            if event_type in config.PASS_LIKE_OPEN + config.INCOMING + config.SET_PIECE + ["tackle"]:
                s, matching_func = ELASTIC._find_matching_func(event_type)
            else:
                continue

            ret = self._window_of_frames(self.events.loc[i], s, event_type == "tackle")
            player_window = ret[1].loc[self.last_matched_frame :].copy()
            ball_window = ret[2].loc[self.last_matched_frame :].copy()
            oppo_window = ret[3].loc[self.last_matched_frame :].copy() if ret[3] is not None else None
            window_args = (ret[0], player_window, ball_window, oppo_window)

            if len(player_window) > 0:
                matched_frame = self._find_matching_frame(matching_func, i, *window_args)[0]
                if matched_frame == matched_frame:
                    self.matched_frames[i] = matched_frame
                    self.last_matched_frame = matched_frame

    def synchronize(self) -> None:
        """
        Applies the ELASTIC synchronization algorithm on the instantiated class.
        """
        kickoff_idx = 0

        for period in self.events["period_id"].unique():
            try:  # Find the kickoff event of the period
                matched_frame = self.detect_kickoff(period=period)
                self.last_matched_frame = matched_frame
                self.matched_frames.loc[kickoff_idx] = matched_frame

            except ValueError:  # If there is no candidate frames for the kickoff, then find the second event
                kickoff_frame = self.frames[self.frames["period_id"] == period].index[0]
                self.last_matched_frame = kickoff_frame
                self.matched_frames.loc[kickoff_idx] = kickoff_frame

                kickoff_idx += 1
                window_args = self._window_of_frames(self.events.loc[kickoff_idx], 5)
                matched_frame = self._find_matching_frame(ELASTIC._detect_pass_like, kickoff_idx, *window_args)[0]

            # Adjust the time bias between events and tracking
            ts_offset = self.events.at[kickoff_idx, "utc_timestamp"] - self.frames.at[matched_frame, "utc_timestamp"]
            self.events.loc[self.events["period_id"] == period, "utc_timestamp"] -= ts_offset
            kickoff_idx = len(self.events[self.events["period_id"] == period])

            # Sync events of the playing period
            self._sync_period_events(period)

        self.events["frame"] = self.matched_frames
        # self.events["seconds"] = (self.matched_frames / self.fps).round(2)

        # Detect receptions
        self.reception_det = ReceptionDetector(self.events, self.tracking)
        self.reception_det.run()
        # self.events = pd.merge(self.events, self.reception_det.events, how="left")
        self.events = self.reception_det.events

        # Post-synchronize remaining events
        post_sync_events = self.events[self.events["spadl_type"].isin(config.MINOR)]
        self.events.loc[post_sync_events.index, "frame"] = np.nan
        self.matched_frames.loc[post_sync_events.index] = np.nan

        for i in tqdm(post_sync_events.index, desc="Post-syncing remaining events"):
            event_type = self.events.at[i, "spadl_type"]
            event_player = self.events.at[i, "player_id"]
            prev_player = self.events.at[i - 1, "player_id"]
            prev_receiver = self.events.at[i - 1, "receiver_id"]

            if event_type in ["take_on", "bad_touch"]:
                prev_receive_frame = self.events.at[i - 1, "receive_frame"]
                if event_player == prev_receiver and not np.isnan(prev_receive_frame):
                    self.matched_frames[i] = prev_receive_frame
                    continue

                prev_frame = self.events.at[i - 1, "frame"]
                if event_type == "take_on" and event_player == prev_player and not np.isnan(prev_frame):
                    self.matched_frames[i] = prev_frame
                    continue

            if event_type == "dispossessed" and self.events.at[i, "next_type"] == "tackle":
                next_frame = self.matched_frames[i + 1]
                if not np.isnan(next_frame):
                    self.matched_frames[i] = next_frame
                    continue

            if event_type == "foul" and self.events.at[i - 1, "spadl_type"] == "foul":
                prev_frame = self.matched_frames[i - 1]
                if not np.isnan(prev_frame):
                    self.matched_frames[i] = prev_frame
                    continue

            prev_frames = self.matched_frames[: i - 1].values
            prev_receive_frames = self.events.loc[: i - 1, "receive_frame"].values
            min_frame = np.nanmax([np.nanmax(prev_frames), np.nanmax(prev_receive_frames), 0])

            next_frames = self.matched_frames[i:].values
            max_frame = np.nanmin([np.nanmin(next_frames), self.frames.index[-1]])

            s, matching_func = ELASTIC._find_matching_func(event_type)
            ret = self._window_of_frames(post_sync_events.loc[i], s)

            player_window = ret[1].loc[min_frame:max_frame].copy()
            ball_window = ret[2].loc[min_frame:max_frame].copy()
            # oppo_window = ret[3].loc[min_frame:max_frame].copy() if ret[3] is not None else None
            window_args = (ret[0], player_window, ball_window)

            if len(player_window) > 0:
                matched_frame = self._find_matching_frame(matching_func, i, *window_args)[0]
                if matched_frame == matched_frame:
                    self.matched_frames[i] = matched_frame

        self.events["frame"] = self.matched_frames
        # self.events["seconds"] = (self.matched_frames / self.fps).round(2)
        self.events["synced_ts"] = self.events["frame"].map(self.frames["timestamp"].to_dict())
        self.events["receive_ts"] = self.events["receive_frame"].map(self.frames["timestamp"].to_dict())

    def plot_window_features(self, event_idx: int, save_path: str = None) -> pd.DataFrame:
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
        prev_frames = self.matched_frames.loc[: event_idx - 1].values
        # prev_receive_frames = self.events.loc[: event_idx - 1, "receive_frame"].values
        # min_frame = np.nanmax([np.nanmax(prev_frames), np.nanmax(prev_receive_frames), 0])
        min_frame = np.nanmax([np.nanmax(prev_frames), 0])

        event = self.events.loc[event_idx]
        event_type = event["spadl_type"]
        s, matching_func = ELASTIC._find_matching_func(event_type)
        print(f"Event {event_idx}: {event_type} by {self.events.at[event_idx, 'player_id']}")

        duel_like = event_type in ["tackle"]
        event_frame, player_window, ball_window, oppo_window = self._window_of_frames(event, s, duel_like)
        player_window = player_window.loc[min_frame:].copy()
        ball_window = ball_window.loc[min_frame:].copy()
        oppo_window = oppo_window.loc[min_frame:].copy() if oppo_window is not None else None
        window_args = (event_frame, player_window, ball_window, oppo_window)

        matched_frame, features, cand_frames = self._find_matching_frame(matching_func, event_idx, *window_args)

        if not pd.isna(matched_frame):
            matched_period = self.frames.at[matched_frame, "period_id"]
            matched_time = self.frames.at[matched_frame, "timestamp"]
            print(f"Matched frame: {matched_frame}")
            print(f"Matched time: P{matched_period}-{matched_time}")

        else:
            period_id = self.events.at[event_idx, "period_id"]
            period_events = self.events[self.events["period_id"] == period_id]
            kickoff_frame = period_events["frame"].iloc[0]
            kickoff_ts = period_events["utc_timestamp"].iloc[0]

            recorded_ts = self.events.at[event_idx, "utc_timestamp"]
            recorded_total_seconds = (recorded_ts - kickoff_ts).total_seconds()
            recorded_frame = int(kickoff_frame + self.fps * recorded_total_seconds)
            print(f"Recorded frame: {recorded_frame}")

            if abs(self.frames.index.values - recorded_frame).min() < self.fps * config.TIME_SET_PIECE:
                recorded_frame = self.frames.index[abs(self.frames.index.values - recorded_frame).argmin()]
                period = self.frames.at[recorded_frame, "period_id"]
                recorded_time = self.frames.at[recorded_frame, "timestamp"]
                print(f"Closest time: P{period}-{recorded_time}")
            else:
                print("Out-of-play at the recorded frame.")

        if features.empty:
            return

        else:
            features["frame_delay"] = features["frame_delay"] / 5
            if event_type in config.INCOMING:
                features["ball_accel"] = features["rel_accel"] / 5
            else:
                features["ball_accel"] = features["ball_accel"] / 5

            if duel_like:
                features_to_plot = ["player_dist", "ball_accel", "oppo_dist"]
            else:
                features_to_plot = ["player_dist", "ball_accel", "frame_delay"]

            plt.rcParams.update({"font.size": 18})
            plt.figure(figsize=(8, 6))
            plt.plot(features[features_to_plot], label=features_to_plot)

            ymax = 25
            plt.ylim(0, ymax)
            plt.vlines(event_frame, 0, ymax, color="k", linestyles="-")

            if isinstance(cand_frames, List):
                for frame in cand_frames:
                    if frame == matched_frame:
                        plt.vlines(frame, 0, ymax, color="red", linestyles="-")
                    else:
                        plt.vlines(frame, 0, ymax, color="black", linestyles="--")

            elif not pd.isna(matched_frame):
                plt.vlines(matched_frame, 0, ymax, color="red", linestyles="-")

            plt.legend(loc="upper right", fontsize=15)
            plt.grid(axis="y")

            if save_path is not None:
                plt.savefig(save_path, bbox_inches="tight")

            plt.show()

        return features
