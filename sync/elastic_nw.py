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
    fps : float
        Tracking data FPS.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        tracking: pd.DataFrame,
        fps: float = 25.0,
        detect_controls: bool = True,
    ) -> None:
        schema.elastic_event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        # Ensure unique indices
        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()
        self.tracking = tracking
        self.fps = fps

        # Define an episode as a sequence of consecutive in-play frames
        time_cols = ["frame_id", "period_id", "timestamp", "utc_timestamp"]
        self.frames = self.tracking[time_cols].drop_duplicates().sort_values("frame_id").set_index("frame_id")
        self.frames["timestamp"] = self.frames["timestamp"].apply(utils.seconds_to_timestamp)
        self.frames["episode_id"] = 0
        n_prev_episodes = 0

        for i in self.tracking["period_id"].unique():
            period_frames = self.frames.loc[self.frames["period_id"] == i].index.values
            episode_ids = (np.diff(period_frames, prepend=-5) >= 5).astype(int).cumsum() + n_prev_episodes
            self.frames.loc[self.frames["period_id"] == i, "episode_id"] = episode_ids
            n_prev_episodes = episode_ids.max()

        if "episode_id" not in self.events.columns:
            self.events = self.find_event_episodes(self.events)

        if detect_controls:
            self.events = ELASTIC_NW.insert_control_events(self.events)
            self.events = ELASTIC_NW.insert_out_events(self.events)

        # Precomputed candidate frames (to be filled by find_candidate_frames)
        self.cand_frames: pd.DataFrame = None

    @staticmethod
    def infer_out_player_id(event: pd.Series) -> str:
        event_type = event.get("spadl_type")
        x = pd.to_numeric(event.get("start_x"), errors="coerce")
        y = pd.to_numeric(event.get("start_y"), errors="coerce")
        if event_type == "throw_in":
            return "out_top" if float(y) >= config.PITCH_Y / 2 else "out_bottom"
        else:
            return "out_left" if float(x) <= config.PITCH_X / 2 else "out_right"

    @staticmethod
    def insert_out_events(events: pd.DataFrame) -> pd.DataFrame:
        """Insert virtual out events before OOP set pieces that end pass-like sequences."""
        assert "episode_id" in events.columns

        events = events.copy()
        prev_events = events.shift(1)
        pass_like_types = config.PASS_LIKE_OPEN + config.SET_PIECE
        target_mask = (
            events["spadl_type"].isin(config.SET_PIECE_OOP)
            & prev_events["spadl_type"].isin(pass_like_types)
            & (prev_events["period_id"] == events["period_id"])
            & (prev_events["episode_id"] != events["episode_id"])
            & (prev_events["utc_timestamp"] < events["utc_timestamp"])
        )
        if not target_mask.any():
            return events

        out_events = events.loc[target_mask].copy()
        out_events["player_id"] = out_events.apply(ELASTIC_NW.infer_out_player_id, axis=1)
        out_events["spadl_type"] = "out"
        out_events["success"] = True
        out_events["episode_id"] = prev_events.loc[target_mask, "episode_id"].to_numpy()
        out_events["utc_timestamp"] = prev_events.loc[target_mask, "utc_timestamp"].to_numpy()

        events["order"] = events.index.astype(float)
        out_events["order"] = out_events.index.astype(float) - 0.5

        combined = pd.concat([events, out_events], axis=0, ignore_index=False)
        combined = combined.sort_values("order", kind="mergesort", ignore_index=True).drop(columns=["order"])
        return combined

    @staticmethod
    def insert_control_events(events: pd.DataFrame) -> None:
        """Insert control (reception) events before pass-like or dispossessed events."""
        assert "episode_id" in events.columns

        prev_events = events.shift(1)
        target_types = ["pass", "cross", "shot", "clearance", "dispossessed"]
        target_mask = (
            (events["spadl_type"].isin(target_types))
            & (prev_events["episode_id"] == events["episode_id"])
            & (prev_events["player_id"] != events["player_id"])
            & (prev_events["utc_timestamp"] < events["utc_timestamp"])
        )

        control_events = events.loc[target_mask].copy()
        control_events["spadl_type"] = "control"
        control_events["success"] = True

        events = events.copy()
        events["order"] = events.index.astype(float)
        control_events["order"] = control_events.index.astype(float) - 0.5

        combined = pd.concat([events, control_events], axis=0, ignore_index=False)
        combined = combined.sort_values("order", kind="mergesort", ignore_index=True).drop(columns=["order"])

        return combined

    def find_event_episodes(self, events: pd.DataFrame) -> pd.DataFrame:
        """Assign the nearest episode to each event based on utc_timestamp.

        Enforces that each episode's first event is a set piece or a pass when possible.
        Returns a copy of self.events with an added 'episode_id' column.
        """
        events = events.copy()
        events["episode_id"] = 0
        allowed_start_types = config.SET_PIECE + ["pass", "cross", "control"]

        for period_id in events["period_id"].dropna().unique():
            period_events: pd.DataFrame = events[events["period_id"] == period_id]
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

            period_episode_ids = self.frames.loc[self.frames["period_id"] == period_id, "episode_id"].unique()
            for episode_id in sorted(period_episode_ids, reverse=True):
                ep_events = events[events["episode_id"] == episode_id].sort_values("utc_timestamp")
                allowed_mask = ep_events["spadl_type"].isin(allowed_start_types).to_numpy()
                if not allowed_mask.any():
                    continue

                first_allowed_pos = int(np.flatnonzero(allowed_mask)[0])
                prev_episode_id = episode_id - 1
                if first_allowed_pos > 0 and prev_episode_id in period_episode_ids:
                    prefix_idx = ep_events.index[:first_allowed_pos]
                    events.loc[prefix_idx, "episode_id"] = prev_episode_id

        return events

    def find_candidate_frames(self, period: int = None) -> pd.DataFrame:
        """Find candidate frames for alignment based on physical constraints.

        A frame is a candidate if:
        1. player-ball distance < 3m
        2. ball height < 3.5m
        3. it is a player-ball distance valley, ball height valley, or ball acceleration peak

        Virtual out nodes (out_left/right/top/bottom) use ball-to-boundary
        distances instead of player-ball distance and are kept only when that
        distance is below 1m.

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
        output_cols = [
            "episode_id",
            "frame_id",
            "player_id",
            "player_dist",
            "ball_height",
            "ball_accel",
            "oppo_id",
            "oppo_dist",
        ]

        if period is None:
            tracking = self.tracking.copy()
            frames = self.frames
        else:
            tracking = self.tracking[self.tracking["period_id"] == period].copy()
            frames = self.frames[self.frames["period_id"] == period]

        if self.tracking.empty or self.frames.empty:
            return pd.DataFrame(columns=output_cols)

        events = self.events.copy()
        if "episode_id" not in events.columns:
            events = self.find_event_episodes(events)

        tracking = tracking.merge(frames["episode_id"], left_on="frame_id", right_index=True, how="inner")
        cand_frames: List[pd.DataFrame] = []

        for episode_id, episode_tracking in tqdm(tracking.groupby("episode_id"), desc="Detecting candidate frames"):
            episode_frames = frames[frames["episode_id"] == episode_id].index.values
            if len(episode_frames) == 0:
                continue

            ball_data = episode_tracking[episode_tracking["ball"]].set_index("frame_id").sort_index()
            if ball_data.empty:
                continue

            ball_heights = ball_data["z"].to_numpy()
            ball_accels = ball_data["accel_v"].to_numpy()
            height_valleys = find_peaks(-ball_heights, prominence=0.5)[0]
            accel_peaks = find_peaks(ball_accels, prominence=10, distance=10)[0]
            height_peak_frames = ball_data.index[height_valleys] if height_valleys.size > 0 else []
            accel_peak_frames = ball_data.index[accel_peaks] if accel_peaks.size > 0 else []

            ball_features = ball_data[["x", "y", "z", "accel_v"]].copy()
            ball_features.columns = ["ball_x", "ball_y", "ball_height", "ball_accel"]
            player_data = episode_tracking[episode_tracking["player_id"].notna()]
            if player_data.empty:
                merged = pd.DataFrame(columns=["frame_id", "player_id", "ball_height", "ball_accel", "player_dist"])
            else:
                merged = player_data.merge(ball_features, left_on="frame_id", right_index=True, how="inner")
                dist_x = merged["x"] - merged["ball_x"]
                dist_y = merged["y"] - merged["ball_y"]
                merged["player_dist"] = np.sqrt(dist_x**2 + dist_y**2)

            episode_cands: List[pd.DataFrame] = []

            def _build_candidate_rows(features: pd.DataFrame, player_id: str) -> pd.DataFrame:
                if features.empty:
                    return pd.DataFrame()

                dist_valleys = find_peaks(-features["player_dist"].values, prominence=1)[0]
                height_pos = features.index.get_indexer(height_peak_frames)
                height_pos = height_pos[height_pos >= 0]
                accel_pos = features.index.get_indexer(accel_peak_frames)
                accel_pos = accel_pos[accel_pos >= 0]

                cand_idx = set(dist_valleys.tolist())
                for i in height_pos:
                    if not any(abs(i - c) <= 3 for c in cand_idx):
                        cand_idx.add(int(i))
                for i in accel_pos:
                    if not any(abs(i - c) <= 3 for c in cand_idx):
                        cand_idx.add(int(i))

                first_pos = features.index.get_indexer([episode_frames[0]])[0]
                if first_pos >= 0:
                    cand_idx.add(int(first_pos))

                last_pos = features.index.get_indexer([episode_frames[-1]])[0]
                if last_pos >= 0:
                    cand_idx.add(int(last_pos))

                if len(cand_idx) == 0:
                    return pd.DataFrame()

                cand_idx = sorted(cand_idx)
                player_cands = features.iloc[cand_idx].copy()
                player_cands["episode_id"] = episode_id
                player_cands["player_id"] = player_id

                for idx in player_cands.index:
                    window_start = max(features.index[0], idx - 3)
                    window_end = min(features.index[-1], idx + 3)
                    window = features.loc[window_start:window_end]

                    player_cands.at[idx, "player_dist"] = window["player_dist"].min()
                    player_cands.at[idx, "ball_height"] = window["ball_height"].min()
                    player_cands.at[idx, "ball_accel"] = window["ball_accel"].max()

                if str(player_id).startswith("out_"):
                    valid_mask = player_cands["player_dist"] < 1
                else:
                    valid_mask = (player_cands["player_dist"] < 3) & (player_cands["ball_height"] < 3.5)
                return player_cands[valid_mask].reset_index()

            for player_id, group in merged.groupby("player_id"):
                features = group.set_index("frame_id")[["player_dist", "ball_height", "ball_accel"]].sort_index()
                player_cands = _build_candidate_rows(features, player_id)
                if not player_cands.empty:
                    episode_cands.append(player_cands)

            out_feature_map = {
                "out_left": ball_data["x"],
                "out_right": config.PITCH_X - ball_data["x"],
                "out_bottom": ball_data["y"],
                "out_top": config.PITCH_Y - ball_data["y"],
            }
            for out_player_id, out_dists in out_feature_map.items():
                out_features = pd.DataFrame(
                    {
                        "player_dist": out_dists,
                        "ball_height": ball_data["z"],
                        "ball_accel": ball_data["accel_v"],
                    }
                )
                out_cands = _build_candidate_rows(out_features, out_player_id)
                if not out_cands.empty:
                    episode_cands.append(out_cands)

            if len(episode_cands) == 0:
                continue

            episode_cands = pd.concat(episode_cands, ignore_index=True)

            first_frame_id = int(episode_frames[0])
            if first_frame_id not in episode_cands["frame_id"].values:
                episode_events = events[events["episode_id"] == episode_id].copy()

                if not episode_events.empty:
                    first_player_id = episode_events["player_id"].iloc[0]
                    mask = (merged["frame_id"] == first_frame_id) & (merged["player_id"] == first_player_id)
                    tracking_row = merged[mask]

                    if not tracking_row.empty:
                        player_dist = float(tracking_row["player_dist"].iloc[0])
                        ball_height = float(tracking_row["ball_height"].iloc[0])
                        ball_accel = float(tracking_row["ball_accel"].iloc[0])
                        first_row = pd.DataFrame(
                            [
                                {
                                    "episode_id": episode_id,
                                    "frame_id": first_frame_id,
                                    "player_id": first_player_id,
                                    "player_dist": player_dist,
                                    "ball_height": ball_height,
                                    "ball_accel": ball_accel,
                                }
                            ]
                        )
                        episode_cands = pd.concat([first_row, episode_cands], ignore_index=True)

            episode_cands = self.calculate_oppo_features(episode_cands, merged_data=merged)
            cand_frames.append(episode_cands[output_cols])

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

        out_mask = output["player_id"].astype(str).str.startswith(("out_", "goal_"))
        output.loc[out_mask & output["pre_kick_dist"].isna(), "pre_kick_dist"] = 5.0
        output.loc[out_mask & output["post_kick_dist"].isna(), "post_kick_dist"] = 5.0

        return output

    def calculate_oppo_features(self, cand_frames: pd.DataFrame, merged_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add opponent features based on players within 3m of the ball in the same frame.

        oppo_id/oppo_dist are taken from the closest opponent (by player_dist) in the same frame_id.
        """
        output = cand_frames.copy()
        output["oppo_id"] = np.nan
        output["oppo_dist"] = np.nan

        if output.empty:
            return output

        if merged_data is None:
            player_data = self.tracking.loc[self.tracking["player_id"].notna(), ["frame_id", "player_id", "x", "y"]]
            ball_data = self.tracking.loc[self.tracking["ball"], ["frame_id", "x", "y"]]
            merged_data = player_data.merge(ball_data.rename(columns={"x": "ball_x", "y": "ball_y"}), on="frame_id")
            if merged_data.empty:
                return output
            dist_x = merged_data["x"] - merged_data["ball_x"]
            dist_y = merged_data["y"] - merged_data["ball_y"]
            merged_data["player_dist"] = np.sqrt(dist_x**2 + dist_y**2)
        else:
            merged_data = merged_data.copy()
            if "player_dist" not in merged_data.columns:
                if {"x", "y", "ball_x", "ball_y"}.issubset(merged_data.columns):
                    dist_x = merged_data["x"] - merged_data["ball_x"]
                    dist_y = merged_data["y"] - merged_data["ball_y"]
                    merged_data["player_dist"] = np.sqrt(dist_x**2 + dist_y**2)
                else:
                    return output

        eligible = merged_data[merged_data["player_dist"] <= 3].copy()
        if eligible.empty:
            return output

        eligible["team"] = eligible["player_id"].astype(str).str[:4]
        eligible = eligible.sort_values(["frame_id", "team", "player_dist"], na_position="last")
        min_by_team = eligible.drop_duplicates(["frame_id", "team"])

        home_min = min_by_team[min_by_team["team"] == "home"][["frame_id", "player_id", "player_dist"]]
        away_min = min_by_team[min_by_team["team"] == "away"][["frame_id", "player_id", "player_dist"]]
        home_min.columns = ["frame_id", "home_id", "home_dist"]
        away_min.columns = ["frame_id", "away_id", "away_dist"]

        output["team"] = output["player_id"].astype(str).str[:4]
        output = output.merge(home_min, on="frame_id", how="left").merge(away_min, on="frame_id", how="left")

        output["oppo_id"] = np.where(
            output["team"] == "home",
            output["away_id"],
            np.where(output["team"] == "away", output["home_id"], np.nan),
        )
        output["oppo_dist"] = np.where(
            output["team"] == "home",
            output["away_dist"],
            np.where(output["team"] == "away", output["home_dist"], np.nan),
        )
        output["oppo_dist"] = output["oppo_dist"].fillna(10.0)

        return output.drop(columns=["team", "home_id", "home_dist", "away_id", "away_dist"])

    def _append_foul_alignment(self, aligned: pd.DataFrame, ep_frames: pd.DataFrame) -> pd.DataFrame:
        aligned_frames = aligned["frame_id"].dropna()
        if aligned_frames.empty:
            return aligned

        episode_id = aligned["episode_id"].iloc[0]
        episode_events: pd.DataFrame = self.events[self.events["episode_id"] == episode_id]
        if episode_events.empty or episode_events["spadl_type"].iloc[-1] != "foul":
            return aligned

        foul_event: pd.Series = episode_events.iloc[-1].copy()
        last_aligned_frame = aligned_frames.iloc[-1]

        player_cands = ep_frames[ep_frames["player_id"] == foul_event["player_id"]]
        if not player_cands.empty:
            last_cand = player_cands.sort_values("frame_id").iloc[-1]
            last_cand_frame = last_cand["frame_id"]
            if pd.notna(last_cand_frame) and last_cand_frame >= last_aligned_frame:
                foul_event["frame_id"] = last_cand_frame
                foul_event["timestamp"] = self.frames.at[last_cand_frame, "timestamp"]
                foul_event["score"] = np.nan
                aligned.loc[foul_event.name] = foul_event[config.ALIGNED_COLS]
                return aligned

        return aligned

    def align_episode(
        self, episode_id: int, events: pd.DataFrame = None, cand_frames: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align events and candidate frames for a single episode using Needleman-Wunsch algorithm.
        PASS_LIKE_OPEN, SET_PIECE, INCOMING, bad_touch, tackle, and dispossessed are aligned.

        Returns
        -------
        aligned : pd.DataFrame
            Alignment result with matched frames.
        score_mat : pd.DataFrame
            Event-frame score matrix (n_events x n_frames) with event indices as rows and frame_ids as columns.
        dp_mat : pd.DataFrame
            DP matrix (n_events+1 x n_frames+1) with -1 as the first index and column.
        path : pd.DataFrame
            Optimal alignment path with event/frame positions, ids, and timestamps.
        """
        if events is None:
            assert isinstance(events, pd.DataFrame)
            events = self.events.copy()

        if "episode_id" not in events.columns:
            events = self.find_event_episodes(events)
        else:
            events = events.copy()
        events = events.drop(columns=["frame_id", "synced_ts", "score"], errors="ignore")

        if cand_frames is None:
            assert isinstance(self.cand_frames, pd.DataFrame)
            cand_frames = self.cand_frames.copy()

        if "pre_kick_dist" not in cand_frames.columns:
            cand_frames = self.calculate_kick_dists(cand_frames)

        minor_types = ["bad_touch", "tackle", "dispossessed"]
        event_types = config.PASS_LIKE_OPEN + config.SET_PIECE + config.INCOMING + minor_types + ["out"]

        ep_events = events[(events["episode_id"] == episode_id) & (events["spadl_type"].isin(event_types))]
        ep_frames = cand_frames[cand_frames["episode_id"] == episode_id]
        ep_frame_ids = np.sort(ep_frames["frame_id"].unique())

        if ep_events.empty or len(ep_frame_ids) == 0:
            aligned = pd.DataFrame(columns=events.columns.tolist() + ["frame_id", "score"])
            score_mat = pd.DataFrame(index=ep_events.index, columns=ep_frame_ids, dtype=float)
            path = pd.DataFrame(columns=["event_pos", "frame_pos", "event_idx", "frame_id", "move"])
            dp_idx = [-1] + ep_events.index.tolist()
            dp_cols = [-1] + ep_frame_ids.tolist()
            dp_mat = pd.DataFrame(
                np.zeros((len(dp_idx), len(dp_cols)), dtype=float),
                index=dp_idx,
                columns=dp_cols,
            )
            return aligned, score_mat, dp_mat, path

        n_events = len(ep_events)
        n_frames = len(ep_frame_ids)
        score_mat = np.zeros((n_events, n_frames), dtype=float)
        frame_pos_map = {frame_id: pos for pos, frame_id in enumerate(ep_frame_ids)}

        for i, event_idx in enumerate(ep_events.index):
            event_player = ep_events.at[event_idx, "player_id"]
            event_type = ep_events.at[event_idx, "spadl_type"]
            if event_type == "tackle":
                kick_dist_col = "pre_kick_dist"
                score_fn = utils.nw_score_minor
            elif event_type == "dispossessed":
                kick_dist_col = "post_kick_dist"
                score_fn = utils.nw_score_minor
            else:
                kick_dist_col = "pre_kick_dist" if event_type in config.INCOMING else "post_kick_dist"
                score_fn = utils.nw_score_major
            player_frames = ep_frames[ep_frames["player_id"] == event_player]

            player_scores = score_fn(player_frames, event_player, kick_dist_col)
            player_scores = pd.Series(player_scores, index=player_frames["frame_id"]).groupby(level=0).max()

            frame_pos = [frame_pos_map[frame_id] for frame_id in player_scores.index]
            score_mat[i, frame_pos] = player_scores.to_numpy()

        gap_event = -10.0
        gap_frame = -10.0
        repeat_penalty = -20.0
        repeat_threshold = 50.0
        dp_mat = np.zeros((n_events + 1, n_frames + 1), dtype=float)
        trace = np.zeros((n_events + 1, n_frames + 1), dtype=np.int8)

        for i in range(1, n_events + 1):
            dp_mat[i, 0] = dp_mat[i - 1, 0] + gap_event
            trace[i, 0] = 1
        for j in range(1, n_frames + 1):
            dp_mat[0, j] = dp_mat[0, j - 1] + gap_frame
            trace[0, j] = 2

        for i in range(1, n_events + 1):
            for j in range(1, n_frames + 1):
                diag = dp_mat[i - 1, j - 1] + score_mat[i - 1, j - 1]
                up_gap = dp_mat[i - 1, j] + gap_event
                left_gap = dp_mat[i, j - 1] + gap_frame
                up_match = -np.inf
                if score_mat[i - 1, j - 1] >= repeat_threshold:
                    up_match = dp_mat[i - 1, j] + score_mat[i - 1, j - 1] + repeat_penalty
                if diag >= up_gap and diag >= left_gap and diag >= up_match:
                    dp_mat[i, j] = diag
                    trace[i, j] = 0
                elif up_match >= up_gap and up_match >= left_gap:
                    dp_mat[i, j] = up_match
                    trace[i, j] = 3
                elif up_gap >= left_gap:
                    dp_mat[i, j] = up_gap
                    trace[i, j] = 1
                else:
                    dp_mat[i, j] = left_gap
                    trace[i, j] = 2

        dp_idx = [-1] + ep_events.index.tolist()
        dp_cols = [-1] + ep_frame_ids.tolist()
        dp_mat = pd.DataFrame(dp_mat, index=dp_idx, columns=dp_cols)

        match_rows = []
        path_rows = []
        i = n_events
        j = n_frames
        while i > 0 or j > 0:
            if i > 0 and j > 0 and trace[i, j] in (0, 3):
                event_pos = i - 1
                frame_pos = j - 1
                matched_frame = ep_frame_ids[frame_pos]
                match_rows.append(
                    {
                        "index": ep_events.index[event_pos],
                        "frame_id": matched_frame,
                        "timestamp": self.frames.at[matched_frame, "timestamp"],
                        "score": score_mat[event_pos, frame_pos],
                    }
                )
                if trace[i, j] == 0:
                    move = "diag"
                    i -= 1
                    j -= 1
                else:
                    move = "repeat"
                    i -= 1
            elif i > 0 and (j == 0 or trace[i, j] == 1):
                event_pos = i - 1
                frame_pos = None
                move = "up"
                i -= 1
            else:
                event_pos = None
                frame_pos = j - 1
                move = "left"
                j -= 1

            timestamp = self.frames.loc[ep_frame_ids[frame_pos], "timestamp"] if frame_pos is not None else None
            path_rows.append(
                {
                    "event_pos": event_pos,
                    "frame_pos": frame_pos,
                    "event_idx": ep_events.index[event_pos] if event_pos is not None else None,
                    "frame_id": ep_frame_ids[frame_pos] if frame_pos is not None else None,
                    "timestamp": timestamp,
                    "move": move,
                }
            )

        match_rows.reverse()
        matches = pd.DataFrame(match_rows).set_index("index")

        path_rows.reverse()
        path = pd.DataFrame(path_rows)

        aligned = pd.concat([events.loc[matches.index], matches], axis=1)[config.ALIGNED_COLS]
        aligned = self._append_foul_alignment(aligned, ep_frames)
        score_mat = pd.DataFrame(score_mat, index=ep_events.index, columns=ep_frame_ids)
        return aligned, score_mat, dp_mat, path

    def run(self, events: pd.DataFrame = None, simplify_one_touch: bool = True) -> pd.DataFrame:
        """
        Runs Needleman-Wunsch alignment across the full match by episode.
        """
        if events is None:
            events = self.events.copy().drop(columns=["frame_id", "synced_ts", "score"], errors="ignore")
        else:
            events = events.copy()

        if self.cand_frames is None:
            self.cand_frames = self.find_candidate_frames()

        if "pre_kick_dist" not in self.cand_frames.columns:
            self.cand_frames = self.calculate_kick_dists(self.cand_frames)

        matches = []
        for episode_id in tqdm(self.frames["episode_id"].unique(), desc="Needleman-Wunsch alignment"):
            episode_matches, _, _, _ = self.align_episode(episode_id, events)
            if not episode_matches.empty:
                matches.append(episode_matches)

        if len(matches) > 0:
            aligned = pd.concat(matches).sort_index()
            events.loc[aligned.index, "frame_id"] = aligned["frame_id"].round()
            events.loc[aligned.index, "score"] = aligned["score"]
            events.loc[events["score"] < 30, "frame_id"] = np.nan
        else:
            aligned = pd.DataFrame(columns=config.ALIGNED_COLS)

        events["timestamp"] = events["frame_id"].map(self.frames["timestamp"].to_dict())

        if simplify_one_touch:
            control_mask = events["spadl_type"] == "control"
            one_touch_mask = (events["frame_id"].shift(-1) == events["frame_id"]) | events["frame_id"].shift(-1).isna()
            events = events.loc[~(control_mask & one_touch_mask)].reset_index(drop=True)
        # return events[config.ALIGNED_COLS + ["start_x", "start_y", "utc_timestamp"]]
        return events[config.ALIGNED_COLS]
