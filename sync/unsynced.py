import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from sync import config, schema, utils


class Unsynced:
    """Fill downstream-required columns without actual event-tracking synchronization.

    - synced_ts/frame_id are derived from utc_timestamp differences against the first
      event of each period (rounded to multiples of 1/fps).
    - next_player_id/next_type are computed per period via shift(-1).
    - For pass-like / set-piece events (the receive-detection targets in
      sync.receive.ReceiveDetector), receiver_id / receive_frame_id / receive_ts are
      filled following ReceiveDetector's team-restriction rules. Other events keep
      these fields as NaN.
    """

    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, fps: int = 25) -> None:
        schema.elastic_event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()
        self.tracking = tracking
        self.fps = fps

        # Build frames table with episode_id (same logic as ReceiveDetector)
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

    def _fill_synced_ts_frame_id(self) -> None:
        step = 1.0 / self.fps
        for p in self.events["period_id"].unique():
            pmask = self.events["period_id"] == p
            base_ts = self.events.loc[pmask, "utc_timestamp"].iloc[0]
            diff_sec = (self.events.loc[pmask, "utc_timestamp"] - base_ts).dt.total_seconds()
            self.events.loc[pmask, "synced_ts"] = np.round(diff_sec / step) * step
        self.events["frame_id"] = (self.events["synced_ts"] * self.fps).round()

    def _fill_next_event_cols(self) -> None:
        for p in self.events["period_id"].unique():
            pmask = self.events["period_id"] == p
            self.events.loc[pmask, "next_player_id"] = self.events.loc[pmask, "player_id"].shift(-1)
            self.events.loc[pmask, "next_type"] = self.events.loc[pmask, "spadl_type"].shift(-1)

    def _episode_id_for_frame(self, frame_id: float) -> float:
        if pd.isna(frame_id):
            return np.nan
        nearest = self.frames.index[self.frames.index.get_indexer([frame_id], method="nearest")[0]]
        return self.frames.at[nearest, "episode_id"]

    def _closest_player_at_frame(self, frame_id: float, passer: str, event_type: str, success: bool) -> str:
        nearest = self.frames.index[self.frames.index.get_indexer([frame_id], method="nearest")[0]]
        frame_tracking = self.tracking[self.tracking["frame_id"] == nearest]
        ball_row = frame_tracking[frame_tracking["ball"]]
        if ball_row.empty:
            return None

        ball_x = float(ball_row["x"].iloc[0])
        ball_y = float(ball_row["y"].iloc[0])
        players = frame_tracking[frame_tracking["player_id"].notna()].copy()
        if players.empty:
            return None

        # Team restriction (receive.py L130-138)
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
        self.events["receive_ts"] = np.nan

        # Map each event to its episode using its frame_id
        self.events["episode_id"] = self.events["frame_id"].apply(self._episode_id_for_frame)

        target_mask = self.events["spadl_type"].isin(pass_like)
        target_idxs = self.events.index[target_mask]

        for idx in target_idxs:
            event_episode = self.events.at[idx, "episode_id"]
            event_type = self.events.at[idx, "spadl_type"]
            passer = self.events.at[idx, "player_id"]
            success = bool(self.events.at[idx, "success"])
            next_type = self.events.at[idx, "next_type"]

            # Find next event in same period
            next_idx = idx + 1 if idx + 1 in self.events.index else None
            next_in_same_episode = (
                next_idx is not None
                and pd.notna(self.events.at[next_idx, "episode_id"])
                and self.events.at[next_idx, "episode_id"] == event_episode
            )

            if next_in_same_episode:
                # Case 2: not the last event in episode
                self.events.at[idx, "receiver_id"] = self.events.at[next_idx, "player_id"]
                self.events.at[idx, "receive_frame_id"] = self.events.at[next_idx, "frame_id"]
                self.events.at[idx, "receive_ts"] = self.events.at[next_idx, "synced_ts"]
            else:
                # Case 1: last event in episode
                if pd.isna(event_episode):
                    continue
                ep_last_frame = float(self.frames[self.frames["episode_id"] == event_episode].index[-1])
                self.events.at[idx, "receive_frame_id"] = ep_last_frame
                # synced_ts at this frame: derive from base utc_timestamp of the period
                period_id = self.events.at[idx, "period_id"]
                base_ts = self.events.loc[self.events["period_id"] == period_id, "utc_timestamp"].iloc[0]
                ep_utc = self.frames.loc[ep_last_frame, "utc_timestamp"]
                step = 1.0 / self.fps
                self.events.at[idx, "receive_ts"] = round((ep_utc - base_ts).total_seconds() / step) * step

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
        self._fill_synced_ts_frame_id()
        self._fill_next_event_cols()
        self._fill_receive_cols()
        self._fill_xy_cols()
        self.events["object_id"] = self.events["player_id"]
        return self.events
