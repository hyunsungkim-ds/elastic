import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from sync import config, schema, utils


class Unsynced:
    """Fill downstream-required columns without actual event-tracking synchronization.

    The output format matches ELASTIC_Greedy/ELASTIC_NW exactly; only the values
    of frame_id / synced_ts / receive_* differ in that they are derived from
    utc_timestamps (not from candidate-frame alignment).
    """

    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, fps: int = 25) -> None:
        schema.elastic_event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()
        self.tracking = tracking
        self.fps = fps

        # Build frames table identically to ELASTIC_Greedy
        time_cols = ["frame_id", "period_id", "timestamp", "utc_timestamp"]
        self.frames = self.tracking[time_cols].drop_duplicates().sort_values("frame_id").set_index("frame_id")
        self.frames["timestamp"] = self.frames["timestamp"].apply(utils.seconds_to_timestamp)
        self.frames["episode_id"] = 0
        n_prev_episodes = 0
        for p in self.events["period_id"].unique():
            period_frames = self.frames.loc[self.frames["period_id"] == p].index.values
            episode_ids = (np.diff(period_frames, prepend=-5) >= 5).astype(int).cumsum() + n_prev_episodes
            self.frames.loc[self.frames["period_id"] == p, "episode_id"] = episode_ids
            n_prev_episodes = episode_ids.max()

    def _fill_frames_and_timestamps(self) -> None:
        # Map each event's utc_timestamp to the nearest tracking frame_id (per period)
        for p in self.events["period_id"].unique():
            pmask = self.events["period_id"] == p
            period_frames = self.frames[self.frames["period_id"] == p]
            if period_frames.empty:
                continue
            frame_utcs = period_frames["utc_timestamp"].to_numpy()
            event_utcs = self.events.loc[pmask, "utc_timestamp"].to_numpy()
            diffs = (event_utcs[:, None] - frame_utcs[None, :]).astype("timedelta64[ms]").astype(np.int64)
            nearest_pos = np.abs(diffs).argmin(axis=1)
            self.events.loc[pmask, "frame_id"] = period_frames.index.to_numpy()[nearest_pos].astype(float)

        self.events["synced_ts"] = self.events["frame_id"].map(self.frames["timestamp"].to_dict())

    def _fill_next_event_cols(self) -> None:
        for p in self.events["period_id"].unique():
            pmask = self.events["period_id"] == p
            self.events.loc[pmask, "next_player_id"] = self.events.loc[pmask, "player_id"].shift(-1)
            self.events.loc[pmask, "next_type"] = self.events.loc[pmask, "spadl_type"].shift(-1)

    def _closest_player_at_frame(self, frame_id: float, passer: str, event_type: str, success: bool) -> str:
        frame_tracking = self.tracking[self.tracking["frame_id"] == frame_id]
        ball_row = frame_tracking[frame_tracking["ball"]]
        if ball_row.empty:
            return None

        ball_x = float(ball_row["x"].iloc[0])
        ball_y = float(ball_row["y"].iloc[0])
        players = frame_tracking[frame_tracking["player_id"].notna()].copy()
        if players.empty:
            return None

        pass_types = ["pass", "cross", "freekick_crossed", "freekick_short"] + config.SET_PIECE_OOP
        if event_type in pass_types and passer is not None:
            if success:
                players = players[(players["player_id"].str[:4] == passer[:4]) & (players["player_id"] != passer)]
            else:
                players = players[players["player_id"].str[:4] != passer[:4]]
        if players.empty:
            return None

        dists = np.sqrt((players["x"] - ball_x) ** 2 + (players["y"] - ball_y) ** 2)
        return players.loc[dists.idxmin(), "player_id"]

    def _fill_receive_cols(self) -> None:
        pass_like = config.PASS_LIKE_OPEN + config.SET_PIECE
        shot_types = ["shot", "shot_freekick", "shot_penalty"]

        self.events["receiver_id"] = None
        self.events["receive_frame_id"] = np.nan
        self.events["receive_ts"] = None

        # Map each event to its episode via frame_id
        self.events["episode_id"] = self.events["frame_id"].map(self.frames["episode_id"].to_dict())

        target_idxs = self.events.index[self.events["spadl_type"].isin(pass_like)]

        for idx in target_idxs:
            event_episode = self.events.at[idx, "episode_id"]
            event_type = self.events.at[idx, "spadl_type"]
            passer = self.events.at[idx, "player_id"]
            success = bool(self.events.at[idx, "success"])
            next_type = self.events.at[idx, "next_type"]

            next_idx = idx + 1 if idx + 1 in self.events.index else None
            next_in_same_episode = (
                next_idx is not None
                and pd.notna(self.events.at[next_idx, "episode_id"])
                and self.events.at[next_idx, "episode_id"] == event_episode
            )

            if next_in_same_episode:
                self.events.at[idx, "receiver_id"] = self.events.at[next_idx, "player_id"]
                self.events.at[idx, "receive_frame_id"] = self.events.at[next_idx, "frame_id"]
                self.events.at[idx, "receive_ts"] = self.events.at[next_idx, "synced_ts"]
            else:
                if pd.isna(event_episode):
                    continue
                ep_last_frame = float(self.frames[self.frames["episode_id"] == event_episode].index[-1])
                self.events.at[idx, "receive_frame_id"] = ep_last_frame
                self.events.at[idx, "receive_ts"] = self.frames.at[ep_last_frame, "timestamp"]

                if next_type in config.SET_PIECE_OOP:
                    self.events.at[idx, "receiver_id"] = "out"
                elif event_type in shot_types and success:
                    self.events.at[idx, "receiver_id"] = "goal"
                else:
                    self.events.at[idx, "receiver_id"] = self._closest_player_at_frame(
                        ep_last_frame, passer, event_type, success
                    )

    def _fill_xy_cols(self) -> None:
        ball_tracking = self.tracking[self.tracking["ball"]].drop_duplicates("frame_id").set_index("frame_id")
        self.events["start_x"] = self.events["frame_id"].map(ball_tracking["x"])
        self.events["start_y"] = self.events["frame_id"].map(ball_tracking["y"])
        self.events["end_x"] = self.events["receive_frame_id"].map(ball_tracking["x"])
        self.events["end_y"] = self.events["receive_frame_id"].map(ball_tracking["y"])

    def run(self) -> pd.DataFrame:
        self._fill_frames_and_timestamps()
        self._fill_next_event_cols()
        self._fill_receive_cols()
        self._fill_xy_cols()
        self.events["object_id"] = self.events["player_id"]
        return self.events
