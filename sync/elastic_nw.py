import os
import sys
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

from sync import config, schema, utils


class ELASTIC_NW:
    """Synchronize event and tracking data using Needleman-Wunsch based global alignment.

    Unlike the greedy ELASTIC algorithm, this finds globally optimal alignment
    between events and candidate frames using Needleman-Wunsch algorithm.

    Parameters
    ----------
    events : pd.DataFrame
        Event data to synchronize, according to schema sync.schema.event_schema.
    tracking : pd.DataFrame
        Tracking data to synchronize, according to schema sync.schema.tracking_schema.
    args : dict, optional
        Configuration arguments including 'fps' and 'post_sync_types'.
    """

    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, args: dict = None) -> None:
        schema.elastic_event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        # Ensure unique indices
        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()
        self.tracking = tracking

        if args is None:
            self.fps = 25
            self.post_sync_types = config.MINOR
        else:
            self.fps = args["fps"]
            self.post_sync_types = args["post_sync_types"]

        self.pre_sync_types = list(set(config.SPADL_TYPES) - set(self.post_sync_types))

        # Define an episode as a sequence of consecutive in-play frames
        time_cols = ["frame_id", "period_id", "timestamp", "utc_timestamp"]
        self.frames = self.tracking[time_cols].drop_duplicates().sort_values("frame_id").set_index("frame_id")
        self.frames["timestamp"] = self.frames["timestamp"].apply(utils.seconds_to_timestamp)
        self.frames["episode_id"] = 0
        n_prev_episodes = 0

        for i in self.events["period_id"].unique():
            period_frames = self.frames.loc[self.frames["period_id"] == i].index.values
            episode_ids = (np.diff(period_frames, prepend=-5) >= 5).astype(int).cumsum() + n_prev_episodes
            self.frames.loc[self.frames["period_id"] == i, "episode_id"] = episode_ids
            n_prev_episodes = episode_ids.max()

        # Store synchronization results
        self.matched_frames = pd.Series(np.nan, index=self.events.index)

        # Precomputed candidate frames (to be filled by find_candidate_frames)
        self.cand_frames: pd.DataFrame = None

    def find_event_episodes(self, events: pd.DataFrame) -> pd.DataFrame:
        """Assign the nearest episode to each event based on utc_timestamp.

        Returns a copy of self.events with an added 'episode_id' column.
        """
        events = events.copy()
        events["episode_id"] = 0

        for period_id in events["period_id"].dropna().unique():
            period_events: pd.DataFrame = events[events["period_id"] == period_id]
            if period_events.empty:
                continue
            period_events_sorted = period_events.sort_values("utc_timestamp").reset_index()

            period_frames = self.frames[self.frames["period_id"] == period_id].sort_values("utc_timestamp")
            if period_frames.empty:
                continue
            aligned = pd.merge_asof(
                period_events_sorted.drop(columns=["episode_id"], errors="ignore"),
                period_frames[["utc_timestamp", "episode_id"]],
                on="utc_timestamp",
                direction="nearest",
            )
            events.loc[aligned["index"], "episode_id"] = aligned["episode_id"].values

        return events

    def find_candidate_frames(self, period: int = None) -> pd.DataFrame:
        """Find candidate frames for alignment based on physical constraints.

        A frame is a candidate if:
        1. player-ball distance < 3m
        2. ball height < 3.5m
        3. it is a player-ball distance valley, ball height valley, or ball acceleration peak

        Candidate detection is performed per (episode, player) to avoid crossing
        discontinuities in tracking data.

        Parameters
        ----------
        period : int, optional
            The playing period to find candidates for. If None, uses all periods.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['episode_id', 'frame_id', 'player_id', 'player_dist', 'ball_height', 'ball_accel']
            containing all candidate frames per player.
        """
        output_cols = ["episode_id", "frame_id", "player_id", "player_dist", "ball_height", "ball_accel"]
        if period is None:
            tracking = self.tracking.copy()
            frames = self.frames
        else:
            tracking = self.tracking[self.tracking["period_id"] == period].copy()
            frames = self.frames[self.frames["period_id"] == period]

        if tracking.empty or frames.empty:
            return pd.DataFrame(columns=output_cols)

        tracking = tracking.merge(frames["episode_id"], left_on="frame_id", right_index=True, how="inner")
        cand_frames: List[pd.DataFrame] = []

        for episode_id, episode_tracking in tqdm(tracking.groupby("episode_id"), desc="Detecting candidate frames"):
            episode_frames = frames[frames["episode_id"] == episode_id].index.values
            if len(episode_frames) == 0:
                continue

            ball_data = episode_tracking[episode_tracking["ball"]].set_index("frame_id")
            if ball_data.empty:
                continue

            players = episode_tracking["player_id"].dropna().unique()
            for player_id in players:
                player_data = episode_tracking[episode_tracking["player_id"] == player_id].set_index("frame_id")

                # Find common frames between player and ball
                common_frames = player_data.index.intersection(ball_data.index)
                if len(common_frames) == 0:
                    continue

                player_subset: pd.DataFrame = player_data.loc[common_frames]
                ball_subset: pd.DataFrame = ball_data.loc[common_frames]

                # Compute features
                player_x = player_subset["x"].values
                player_y = player_subset["y"].values
                ball_x = ball_subset["x"].values
                ball_y = ball_subset["y"].values

                player_dists = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
                ball_heights = ball_subset["z"].values
                ball_accels = ball_subset["accel_v"].values

                # Create features DataFrame for peak detection
                features = pd.DataFrame(
                    {
                        "episode_id": episode_id,
                        "frame_id": common_frames,
                        "player_id": player_id,
                        "player_dist": player_dists,
                        "ball_height": ball_heights,
                        "ball_accel": ball_accels,
                    }
                ).set_index("frame_id")
                features = features.sort_index()

                # Find candidate indices (peaks and valleys)
                # 1. Distance valleys (player approaching the ball)
                dist_valleys = find_peaks(-features["player_dist"].values, prominence=1)[0]

                # 2. Height valleys (ball touching the ground or a player)
                height_valleys = find_peaks(-features["ball_height"].values, prominence=0.5)[0]

                # 3. Acceleration peaks (ball being kicked)
                accel_peaks = find_peaks(features["ball_accel"].values, prominence=10, distance=10)[0]

                # Combine all candidate indices (avoiding duplicates within ±3 frames)
                candidates = set(dist_valleys.tolist())

                for i in height_valleys:
                    if not any(abs(i - c) <= 3 for c in candidates):
                        candidates.add(i)

                for i in accel_peaks:
                    if not any(abs(i - c) <= 3 for c in candidates):
                        candidates.add(i)

                # Always include the first frame of each episode (if present in features)
                first_pos = features.index.get_indexer([episode_frames[0]])[0]
                if first_pos >= 0:
                    candidates.add(int(first_pos))

                if len(candidates) == 0:
                    continue

                candidates = sorted(candidates)
                cand_features = features.iloc[candidates].copy()

                # Apply smoothing: use min/max within ±3 frame window
                for idx in cand_features.index:
                    window_start = max(features.index[0], idx - 3)
                    window_end = min(features.index[-1], idx + 3)
                    window = features.loc[window_start:window_end]

                    cand_features.at[idx, "player_dist"] = window["player_dist"].min()
                    cand_features.at[idx, "ball_height"] = window["ball_height"].min()
                    cand_features.at[idx, "ball_accel"] = window["ball_accel"].max()

                # Filter by physical constraints
                valid_mask = (cand_features["player_dist"] < 3) & (cand_features["ball_height"] < 3.5)
                valid_candidates = cand_features[valid_mask].reset_index()

                if len(valid_candidates) > 0:
                    cand_frames.append(valid_candidates)

        if len(cand_frames) == 0:
            return pd.DataFrame(columns=output_cols)
        else:
            cand_frames = pd.concat(cand_frames, ignore_index=True)
            return cand_frames.sort_values(["frame_id", "player_id"]).reset_index(drop=True)

    def calculate_kick_dists(self, cand_frames: pd.DataFrame) -> pd.DataFrame:
        """
        Add pre/post kick distances for each candidate frame within an episode.

        pre_kick_dist: max player_dist from previous candidate (or episode start) to current frame.
        post_kick_dist: max player_dist from current frame to next candidate (or episode end).
        """
        output = cand_frames.copy()
        output["pre_kick_dist"] = np.nan
        output["post_kick_dist"] = np.nan

        if output.empty:
            return output

        player_data = self.tracking.loc[self.tracking["player_id"].notna(), ["frame_id", "player_id", "x", "y"]]
        ball_data = self.tracking.loc[self.tracking["ball"], ["frame_id", "x", "y"]]
        merged = player_data.merge(ball_data.rename(columns={"x": "ball_x", "y": "ball_y"}))
        if merged.empty:
            return output

        dist_x = merged["x"] - merged["ball_x"]
        dist_y = merged["y"] - merged["ball_y"]
        merged["player_dist"] = np.sqrt(dist_x**2 + dist_y**2)
        player_dist_map = {
            player_id: group.set_index("frame_id")["player_dist"].sort_index()
            for player_id, group in merged.groupby("player_id")
        }

        episode_bounds = self.frames.reset_index().groupby("episode_id")["frame_id"].agg(["min", "max"])

        for (episode_id, player_id), group in output.groupby(["episode_id", "player_id"]):
            if player_id not in player_dist_map or episode_id not in episode_bounds.index:
                continue

            player_dists = player_dist_map[player_id]
            episode_start = episode_bounds.at[episode_id, "min"]
            episode_end = episode_bounds.at[episode_id, "max"]
            group_sorted = group.sort_values("frame_id")
            frames = group_sorted["frame_id"].values

            for i, frame in enumerate(frames):
                prev_frame = frames[i - 1] if i > 0 else episode_start
                next_frame = frames[i + 1] if i < len(frames) - 1 else episode_end
                output.at[group_sorted.index[i], "pre_kick_dist"] = player_dists.loc[prev_frame:frame].max()
                output.at[group_sorted.index[i], "post_kick_dist"] = player_dists.loc[frame:next_frame].max()

        return output

    def align_episode(self, cand_frames: pd.DataFrame, episode_id: int) -> pd.DataFrame:
        """
        Align events and candidate frames for a single episode using Needleman-Wunsch algorithm.
        Only PASS_LIKE_OPEN, SET_PIECE, and INCOMING event types are aligned.
        """
        if "episode_id" not in self.events.columns:
            events = self.find_event_episodes(self.events)
        else:
            events = self.events.copy()

        if "pre_kick_dist" not in cand_frames.columns or "post_kick_dist" not in cand_frames.columns:
            cand_frames = self.calculate_kick_dists(cand_frames)

        event_types = config.PASS_LIKE_OPEN + config.SET_PIECE + config.INCOMING
        ep_events = events[(events["episode_id"] == episode_id) & (events["spadl_type"].isin(event_types))]
        ep_frames = cand_frames[cand_frames["episode_id"] == episode_id]
        if ep_events.empty or ep_frames.empty:
            return pd.DataFrame(columns=events.columns.tolist() + ["frame_id", "score"])

        ep_events = ep_events.sort_values("utc_timestamp")
        ep_frames = ep_frames.sort_values("frame_id").reset_index(drop=True)

        event_players = ep_events["player_id"].to_numpy()
        event_types = ep_events["spadl_type"].to_numpy()
        frame_ids = ep_frames["frame_id"].to_numpy()

        n_events = len(ep_events)
        n_frames = len(ep_frames)
        scores = np.zeros((n_events, n_frames), dtype=float)

        for i in range(n_events):
            kick_dist_col = "pre_kick_dist" if event_types[i] in config.INCOMING else "post_kick_dist"
            scores[i, :] = ep_frames.apply(utils.score_nw, args=(event_players[i], kick_dist_col), axis=1).to_numpy()

        gap_event = -10.0
        gap_frame = -10.0
        dp = np.zeros((n_events + 1, n_frames + 1), dtype=float)
        trace = np.zeros((n_events + 1, n_frames + 1), dtype=np.int8)

        for i in range(1, n_events + 1):
            dp[i, 0] = dp[i - 1, 0] + gap_event
            trace[i, 0] = 1
        for j in range(1, n_frames + 1):
            dp[0, j] = dp[0, j - 1] + gap_frame
            trace[0, j] = 2

        for i in range(1, n_events + 1):
            for j in range(1, n_frames + 1):
                diag = dp[i - 1, j - 1] + scores[i - 1, j - 1]
                up = dp[i - 1, j] + gap_event
                left = dp[i, j - 1] + gap_frame
                if diag >= up and diag >= left:
                    dp[i, j] = diag
                    trace[i, j] = 0
                elif up >= left:
                    dp[i, j] = up
                    trace[i, j] = 1
                else:
                    dp[i, j] = left
                    trace[i, j] = 2

        matches = []
        i = n_events
        j = n_frames
        while i > 0 and j > 0:
            if trace[i, j] == 0:
                matched_frame = frame_ids[j - 1]
                matches.append(
                    {
                        "index": ep_events.index[i - 1],
                        "frame_id": matched_frame,
                        "timestamp": self.frames.at[matched_frame, "timestamp"],
                        "score": scores[i - 1, j - 1],
                    }
                )
                i -= 1
                j -= 1
            elif trace[i, j] == 1:
                i -= 1
            else:
                j -= 1

        matches.reverse()
        matches = pd.DataFrame(matches).set_index("index")

        return pd.concat([events.loc[matches.index], matches], axis=1)[config.NW_COLS]

    def run(self) -> pd.DataFrame:
        """
        Runs NW alignment across the full match by episode.
        """
        if "episode_id" not in self.events.columns:
            self.events = self.find_event_episodes(self.events)

        if self.cand_frames is None:
            self.cand_frames = self.find_candidate_frames()
            self.cand_frames = self.calculate_kick_dists(self.cand_frames)

        matches = []
        for episode_id in tqdm(self.frames["episode_id"].unique(), desc="Needleman-Wunsch alignment"):
            episode_matches = self.align_episode(self.cand_frames, episode_id)
            if not episode_matches.empty:
                matches.append(episode_matches)

        if len(matches) > 0:
            aligned = pd.concat(matches).sort_index()
            self.matched_frames.loc[aligned.index] = aligned["frame_id"]
        else:
            aligned = pd.DataFrame(columns=config.NW_COLS)

        self.events["frame_id"] = self.matched_frames
        self.events["synced_ts"] = self.events["frame_id"].map(self.frames["timestamp"].to_dict())

        return aligned
