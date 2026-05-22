import os
import sys
from typing import Dict, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle
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
            self.events = self.insert_out_events(self.events)
            self.events = ELASTIC_NW.insert_goal_events(self.events)

        self.cand_frames: pd.DataFrame = None
        self.score_mats: Dict[int, pd.DataFrame] = {}
        self.dp_mats: Dict[int, pd.DataFrame] = {}
        self.trace_mats: Dict[int, pd.DataFrame] = {}
        self.paths: Dict[int, pd.DataFrame] = {}
        self.synced_events: pd.DataFrame = None

    def _infer_out_player_id(self, event: pd.Series) -> str:
        event_type = event.get("spadl_type")
        team = str(event.get("player_id", ""))[:4]
        is_home = team == "home"

        if event_type == "throw_in":
            utc = event.get("utc_timestamp")
            if pd.notna(utc):
                ball = self.tracking[self.tracking["ball"]].copy()
                time_diff = (ball["utc_timestamp"] - utc).abs()
                nearest_idx = time_diff.idxmin()
                ball_y = ball.at[nearest_idx, "y"]
                return "out_top" if ball_y >= config.PITCH_Y / 2 else "out_bottom"
            return "out_bottom"
        elif event_type == "goalkick":
            return "out_left" if is_home else "out_right"
        else:  # corner_short, corner_crossed
            return "out_right" if is_home else "out_left"

    def insert_out_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Insert virtual out events before OOP set pieces."""
        assert "episode_id" in events.columns

        events = events.copy()
        target_mask = events["spadl_type"].isin(config.SET_PIECE_OOP)

        out_events = events.loc[target_mask].copy()
        out_events["player_id"] = out_events.apply(self._infer_out_player_id, axis=1)
        out_events["spadl_type"] = "out"
        out_events["success"] = True
        out_events["episode_id"] = events.shift(1).loc[target_mask, "episode_id"].to_numpy()
        out_events["utc_timestamp"] = events.shift(1).loc[target_mask, "utc_timestamp"].to_numpy()

        events["order"] = events.index.astype(float)
        out_events["order"] = out_events.index.astype(float) - 0.5

        combined = pd.concat([events, out_events], axis=0, ignore_index=False)
        combined = combined.sort_values("order", kind="mergesort", ignore_index=True).drop(columns=["order"])
        return combined

    @staticmethod
    def insert_goal_events(events: pd.DataFrame) -> pd.DataFrame:
        """Insert virtual goal events after successful shots."""
        events = events.copy()
        shot_types = {"shot", "shot_freekick", "shot_penalty"}
        target_mask = events["spadl_type"].isin(shot_types) & events["success"]
        if not target_mask.any():
            return events

        goal_events = events.loc[target_mask].copy()
        is_home = goal_events["player_id"].str[:4] == "home"
        goal_events["player_id"] = np.where(is_home, "goal_right", "goal_left")
        goal_events["spadl_type"] = "goal"
        goal_events["success"] = True

        events["order"] = events.index.astype(float)
        goal_events["order"] = goal_events.index.astype(float) + 0.5

        combined = pd.concat([events, goal_events], axis=0, ignore_index=False)
        combined = combined.sort_values("order", kind="mergesort", ignore_index=True).drop(columns=["order"])
        return combined

    @staticmethod
    def insert_control_events(events: pd.DataFrame) -> None:
        """Insert control (reception) events before pass-like or dispossessed events."""
        assert "episode_id" in events.columns

        prev_events = events.shift(1)
        target_types = ["pass", "cross", "shot", "clearance", "take_on", "dispossessed"]
        target_mask = (
            (events["spadl_type"].isin(target_types))
            & (prev_events["episode_id"] == events["episode_id"])
            & (prev_events["player_id"] != events["player_id"])
            & (prev_events["utc_timestamp"] < events["utc_timestamp"])
        )

        control_events = events.loc[target_mask].copy()
        control_events["spadl_type"] = "control"
        control_events["success"] = True

        # Also insert control before foul events (preceded by non-shot pass-like or set-piece)
        foul_mask = (
            (events["spadl_type"] == "foul")
            & (prev_events["spadl_type"].isin(["pass", "cross"] + [t for t in config.SET_PIECE if t[:4] != "shot"]))
            & (prev_events["episode_id"] == events["episode_id"])
            & (prev_events["utc_timestamp"] < events["utc_timestamp"])
        )
        if foul_mask.any():
            foul_controls = events.loc[foul_mask].copy()
            foul_controls["spadl_type"] = "control"
            foul_controls["success"] = True

            for idx in foul_controls.index:
                next_idx = idx + 1 if idx + 1 < len(events) else None
                next_is_foul = next_idx is not None and events.at[next_idx, "spadl_type"] == "foul"
                foul_player = events.at[idx, "player_id"]

                if not next_is_foul:
                    # Single foul: control player = foul player
                    foul_controls.at[idx, "player_id"] = foul_player
                else:
                    # Double foul: pick the receiver based on prev pass success
                    prev_player = prev_events.at[idx, "player_id"]
                    next_foul_player = events.at[next_idx, "player_id"]
                    if prev_events.at[idx, "success"]:
                        # Receiver is on the same team as passer
                        receiver = foul_player if prev_player[:4] == foul_player[:4] else next_foul_player
                    else:
                        # Receiver is on the opposite team
                        receiver = foul_player if prev_player[:4] != foul_player[:4] else next_foul_player
                    foul_controls.at[idx, "player_id"] = receiver

            control_events = pd.concat([control_events, foul_controls], ignore_index=False)

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

    def find_candidate_frames(self, period: int = None, slope_window: int = 5) -> pd.DataFrame:
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
            "timestamp",
            "player_id",
            "player_dist",
            "ball_height",
            "ball_accel",
            "pre_slope",
            "post_slope",
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

            ball_accels = ball_data["accel_v"].to_numpy()
            accel_peaks = find_peaks(ball_accels, distance=3, prominence=10)[0]
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

                is_out = str(player_id).startswith("out_")

                # Detect player_dist valleys
                dist_arr = features["player_dist"].to_numpy()
                dist_valleys = find_peaks(-dist_arr, height=-3, distance=3, prominence=0.3)[0]
                cand_idx = set(dist_valleys.tolist())

                # Add ball accel peaks not already covered within ±3 frames
                accel_pos = features.index.get_indexer(accel_peak_frames)
                accel_pos = accel_pos[accel_pos >= 0]
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

                # Calculate pre/post slope using vectorized lookup
                dist_series = features["player_dist"]
                cand_frames_arr = player_cands.index.to_numpy()
                pre_pos = np.searchsorted(dist_series.index, cand_frames_arr - slope_window)
                post_pos = np.searchsorted(dist_series.index, cand_frames_arr + slope_window, side="right") - 1
                pre_dist = dist_series.iloc[np.clip(pre_pos, 0, len(dist_series) - 1)].to_numpy()
                post_dist = dist_series.iloc[np.clip(post_pos, 0, len(dist_series) - 1)].to_numpy()
                cur_dist = dist_series.loc[cand_frames_arr].to_numpy()
                player_cands["pre_slope"] = (cur_dist - pre_dist) / slope_window
                player_cands["post_slope"] = (post_dist - cur_dist) / slope_window

                if is_out:
                    valid_mask = player_cands["player_dist"] < 1
                else:
                    valid_mask = (player_cands["player_dist"] < 3) & (player_cands["ball_height"] < 3.5)
                player_cands = player_cands[valid_mask]

                # Drop flat-slope candidates that have a neighbor within ±10 frames
                flat_mask = (player_cands["post_slope"] - player_cands["pre_slope"]).abs() < 0.05
                frames_arr = player_cands.index.to_numpy()
                left_dist = np.full(len(frames_arr), np.inf)
                right_dist = np.full(len(frames_arr), np.inf)
                left_dist[1:] = frames_arr[1:] - frames_arr[:-1]
                right_dist[:-1] = frames_arr[1:] - frames_arr[:-1]
                min_neighbor_dist = np.minimum(left_dist, right_dist)
                player_cands = player_cands[~(flat_mask.to_numpy() & (min_neighbor_dist <= 20))]

                return player_cands.reset_index()

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
                "goal_left": ball_data["x"],
                "goal_right": config.PITCH_X - ball_data["x"],
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

            # Merge consecutive frames into their group mean
            unique_frames = np.sort(episode_cands["frame_id"].unique()).astype(int)
            gaps = np.diff(unique_frames, prepend=unique_frames[0] - 2) > 1
            group_ids = np.cumsum(gaps)
            group_means = pd.Series(unique_frames).groupby(group_ids).transform("mean")
            frame_mapping = pd.Series(np.round(group_means.values).astype(int), index=unique_frames)
            episode_cands["frame_id"] = episode_cands["frame_id"].map(frame_mapping)

            episode_cands = self.calculate_oppo_features(episode_cands, merged_data=merged)
            episode_cands["timestamp"] = episode_cands["frame_id"].map(frames["timestamp"])
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
                prev_frame = max(frames[i - 1] if i > 0 else episode_start, frame - 50)
                next_frame = min(frames[i + 1] if i < len(frames) - 1 else episode_end, frame + 50)
                output.at[group_sorted.index[i], "pre_kick_dist"] = player_dists.loc[prev_frame:frame].max()
                output.at[group_sorted.index[i], "post_kick_dist"] = player_dists.loc[frame:next_frame].max()

        out_mask = output["player_id"].astype(str).str.startswith(("out_", "goal_"))
        output.loc[out_mask & output["pre_kick_dist"].isna(), "pre_kick_dist"] = 5.0
        output.loc[out_mask & output["post_kick_dist"].isna(), "post_kick_dist"] = 5.0

        return output

    def calculate_takeon_features(self, cand_frames: pd.DataFrame) -> pd.DataFrame:
        """Add take-on scoring features to candidate frames.

        Computes ``player_speed``, ``max_speed``, ``delta_speed``, and
        ``angle_change`` for each candidate frame. These are required by
        ``utils.nw_score_takeon``.

        ``angle_change`` is the cosine of the angle between the opponent's
        relative position vector 0.2 s before and 1 s after the frame,
        capturing how much the dribbler changed direction around the opponent.
        """
        output = cand_frames.copy()
        output[["player_speed", "max_speed", "delta_speed", "angle_change"]] = np.nan

        player_tracking = self.tracking[self.tracking["player_id"].notna()][
            ["frame_id", "player_id", "x", "y", "speed"]
        ]
        # Build per-player lookup indexed by frame_id
        player_xy_speed = {
            pid: grp.set_index("frame_id")[["x", "y", "speed"]].sort_index()
            for pid, grp in player_tracking.groupby("player_id")
        }

        for (episode_id, player_id), group in output.groupby(["episode_id", "player_id"]):
            if str(player_id).startswith("out_"):
                continue
            if player_id not in player_xy_speed:
                continue

            ep_frame_ids = self.frames[self.frames["episode_id"] == episode_id].index
            pt = player_xy_speed[player_id]
            pt = pt[pt.index.isin(ep_frame_ids)]
            if pt.empty:
                continue

            for idx in group.index:
                frame = output.at[idx, "frame_id"]
                if frame not in pt.index:
                    continue

                speed = pt.at[frame, "speed"]
                max_spd = pt.loc[frame : frame + int(0.5 * self.fps), "speed"].max()
                output.at[idx, "player_speed"] = speed
                output.at[idx, "max_speed"] = max_spd
                output.at[idx, "delta_speed"] = max_spd - speed

                oppo_id = output.at[idx, "oppo_id"]
                if pd.isna(oppo_id) or oppo_id not in player_xy_speed:
                    continue

                ot = player_xy_speed[oppo_id]
                ot = ot[ot.index.isin(ep_frame_ids)]
                if ot.empty:
                    continue

                f_before = ot.index[max(ot.index.searchsorted(frame - int(0.2 * self.fps)), 0)]
                f_after = ot.index[min(ot.index.searchsorted(frame + int(self.fps)), len(ot) - 1)]

                if f_before not in pt.index or f_after not in pt.index:
                    continue

                vec1 = (ot.loc[f_before, ["x", "y"]] - pt.loc[f_before, ["x", "y"]]).to_numpy()
                vec2 = (ot.loc[f_after, ["x", "y"]] - pt.loc[f_after, ["x", "y"]]).to_numpy()
                n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                output.at[idx, "angle_change"] = np.dot(vec1, vec2) / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

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

    def _sync_fouls(self, aligned: pd.DataFrame, ep_frames: pd.DataFrame) -> pd.DataFrame:
        if aligned.empty or ep_frames.empty:
            return aligned

        episode_id = aligned["episode_id"].iloc[0]
        episode_events: pd.DataFrame = self.events[self.events["episode_id"] == episode_id]
        if episode_events.empty or (episode_events["spadl_type"] != "foul").all():
            return aligned

        ep_last_frame = self.frames[self.frames["episode_id"] == episode_id].index[-1]
        cands_sorted = ep_frames.sort_values("frame_id")

        foul_pos = np.where(episode_events["spadl_type"].to_numpy() == "foul")[0]
        group_ids = np.cumsum(np.diff(foul_pos, prepend=foul_pos[0] - 2) > 1)
        foul_groups = [foul_pos[group_ids == g].tolist() for g in np.unique(group_ids)]

        for group in foul_groups:
            first_idx = episode_events.index[group[0]]
            last_idx = episode_events.index[group[-1]]

            prev_frames = aligned.loc[aligned.index < first_idx, "frame_id"].dropna()
            prev_frame = prev_frames.iloc[-1] if not prev_frames.empty else 0
            next_frames = aligned.loc[aligned.index > last_idx, "frame_id"].dropna()
            next_frame = next_frames.iloc[0] if not next_frames.empty else ep_last_frame

            window = cands_sorted[(cands_sorted["frame_id"] > prev_frame) & (cands_sorted["frame_id"] <= next_frame)]

            if len(group) == 2:
                player_a = episode_events.iloc[group[0]]["player_id"]
                player_b = episode_events.iloc[group[1]]["player_id"]
                pair = {player_a, player_b}
                match = window[window.apply(lambda r: {r["player_id"], r["oppo_id"]} == pair, axis=1)]
            else:
                player_a = episode_events.iloc[group[0]]["player_id"]
                match = window[window["player_id"] == player_a]

            if not match.empty:
                frame_id = match["frame_id"].iloc[-1]
                for pos in group:
                    foul_event = episode_events.iloc[pos].copy()
                    foul_event["frame_id"] = frame_id
                    foul_event["timestamp"] = self.frames.at[frame_id, "timestamp"]
                    foul_event["score"] = np.nan
                    aligned.loc[foul_event.name] = foul_event[config.ALIGNED_COLS]

        return aligned

    @staticmethod
    def _adjust_dispossessed_frames(aligned: pd.DataFrame) -> pd.DataFrame:
        if len(aligned) < 2:
            return aligned

        types = aligned["spadl_type"].to_numpy()
        players = aligned["player_id"].to_numpy()
        scores = aligned["score"].to_numpy(dtype=float)
        frames = aligned["frame_id"].to_numpy(dtype=float)

        frame_col = aligned.columns.get_loc("frame_id")
        ts_col = aligned.columns.get_loc("timestamp")

        for i in range(len(aligned) - 1):
            # Unify dispossessed-tackle across teams
            if {types[i], types[i + 1]} == {"dispossessed", "tackle"} and players[i][:4] != players[i + 1][:4]:
                better = i if scores[i] >= scores[i + 1] else i + 1
                aligned.iloc[i, frame_col] = frames[better]
                aligned.iloc[i + 1, frame_col] = frames[better]
                aligned.iloc[i, ts_col] = aligned.iloc[better]["timestamp"]
                aligned.iloc[i + 1, ts_col] = aligned.iloc[better]["timestamp"]

            # Snap dispossessed to the next event's frame if same player
            elif types[i] == "dispossessed" and players[i] == players[i + 1]:
                aligned.iloc[i, frame_col] = aligned.iloc[i + 1, frame_col]
                aligned.iloc[i, ts_col] = aligned.iloc[i + 1, ts_col]

        return aligned

    def align_episode(
        self,
        episode_id: int,
        events: pd.DataFrame = None,
        cand_frames: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align events and candidate frames for a single episode using Needleman-Wunsch algorithm.
        PASS_LIKE_OPEN, SET_PIECE, INCOMING, bad_touch, tackle, dispossessed, and optionally
        take_on are aligned.

        Returns
        -------
        aligned : pd.DataFrame
            Alignment result with matched frames.
        score_mat : pd.DataFrame
            Event-frame score matrix (n_events x n_frames) with event indices as rows and frame_ids as columns.
        dp_mat : pd.DataFrame
            DP matrix (n_events+1 x n_frames+1) with -1 as the first index and column.
        trace : pd.DataFrame
            Backpointer matrix with the same shape/index/columns as dp_mat. Values:
            0=diag-match, 1=down-gap, 2=right-gap, 3=down-match.
        path : pd.DataFrame
            Optimal alignment path with event/frame positions, ids, and timestamps.
        """
        if events is None:
            # assert isinstance(events, pd.DataFrame)
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

        if "player_speed" not in cand_frames.columns:
            cand_frames = self.calculate_takeon_features(cand_frames)

        minor_types = ["bad_touch", "tackle", "dispossessed", "take_on"]
        event_types = config.PASS_LIKE_OPEN + config.SET_PIECE + config.INCOMING + minor_types + ["out", "goal"]

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
            trace = pd.DataFrame(
                np.zeros((len(dp_idx), len(dp_cols)), dtype=np.int8),
                index=dp_idx,
                columns=dp_cols,
            )
            return aligned, score_mat, dp_mat, trace, path

        n_events = len(ep_events)
        n_frames = len(ep_frame_ids)
        score_mat = np.zeros((n_events, n_frames), dtype=float)
        frame_pos_map = {frame_id: pos for pos, frame_id in enumerate(ep_frame_ids)}

        for i, event_idx in enumerate(ep_events.index):
            event_player = ep_events.at[event_idx, "player_id"]
            event_type = ep_events.at[event_idx, "spadl_type"]
            is_incoming = event_type in config.INCOMING + ["bad_touch", "tackle"]
            if event_type == "tackle":
                score_fn = utils.nw_score_minor
            elif event_type == "dispossessed":
                score_fn = utils.nw_score_minor
            elif event_type == "take_on":
                score_fn = utils.nw_score_takeon
            else:
                score_fn = utils.nw_score_major
            player_frames = ep_frames[ep_frames["player_id"] == event_player]

            player_scores = score_fn(player_frames, event_player, incoming=is_incoming)
            player_scores = pd.Series(player_scores, index=player_frames["frame_id"]).groupby(level=0).max()

            frame_pos = [frame_pos_map[frame_id] for frame_id in player_scores.index]
            score_mat[i, frame_pos] = player_scores.to_numpy()

        gap_event = 0.1
        gap_frame = 0.0
        repeat_threshold = 0.5

        event_players = ep_events["player_id"].to_numpy()
        event_types = ep_events["spadl_type"].to_numpy()

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
                    if event_players[i - 1] == event_players[i - 2]:
                        if event_types[i - 1] in [event_types[i - 2], "bad_touch"]:
                            repeat_penalty = -0.3
                        elif event_types[i - 1] == "ball_recovery":
                            repeat_penalty = -0.1
                        else:
                            repeat_penalty = 0.0
                    else:
                        repeat_penalty = -0.1
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
        if match_rows:
            matches = pd.DataFrame(match_rows).set_index("index")
        else:
            matches = pd.DataFrame(columns=["frame_id", "timestamp", "score"], index=pd.Index([], name="index"))

        path_rows.reverse()
        path = pd.DataFrame(path_rows)

        aligned = pd.concat([events.loc[matches.index], matches], axis=1)[config.ALIGNED_COLS]
        aligned = self._sync_fouls(aligned, ep_frames)
        aligned = self._adjust_dispossessed_frames(aligned)
        score_mat = pd.DataFrame(score_mat, index=ep_events.index, columns=ep_frame_ids)
        trace = pd.DataFrame(trace, index=dp_idx, columns=dp_cols)
        return aligned, score_mat, dp_mat, trace, path

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

        if "player_speed" not in self.cand_frames.columns:
            self.cand_frames = self.calculate_takeon_features(self.cand_frames)

        matches = []
        self.score_mats = {}
        self.dp_mats = {}
        self.trace_mats = {}
        self.paths = {}
        for episode_id in tqdm(self.frames["episode_id"].unique(), desc="Needleman-Wunsch alignment"):
            episode_matches, score_mat, dp_mat, trace, path = self.align_episode(episode_id, events)
            self.score_mats[episode_id] = score_mat
            self.dp_mats[episode_id] = dp_mat
            self.trace_mats[episode_id] = trace
            self.paths[episode_id] = path
            if not episode_matches.empty:
                matches.append(episode_matches)

        if len(matches) > 0:
            aligned = pd.concat(matches).sort_index()
            events.loc[aligned.index, "frame_id"] = aligned["frame_id"].round()
            events.loc[aligned.index, "score"] = aligned["score"]
            events.loc[events["score"] < 0.3, "frame_id"] = np.nan
        else:
            aligned = pd.DataFrame(columns=config.ALIGNED_COLS)

        events["synced_ts"] = events["frame_id"].map(self.frames["timestamp"].to_dict())

        # Preserve original-indexed events so score_mat/dp_mat row indices line up.
        self.synced_events = events.copy()

        if simplify_one_touch:
            control_mask = events["spadl_type"] == "control"
            one_touch_mask = (events["frame_id"].shift(-1) == events["frame_id"]) | events["frame_id"].shift(-1).isna()
            events = events.loc[~(control_mask & one_touch_mask)].reset_index(drop=True)

        return events

    def _slice_episode(self, start_frame: int, end_frame: int) -> Tuple[int, list, list]:
        """Pick the episode covering ``start_frame`` and return (episode_id, event_rows, frame_cols).

        Episode selection: the one whose frame range includes ``start_frame``;
        otherwise the next episode starting at or after ``start_frame``. Returns
        ``(None, [], [])`` if no episode matches. ``frame_cols`` is NOT filtered
        by zero scores — callers can drop them as needed.
        """
        assert self.synced_events is not None, "Call run() first to populate matrices."

        episode_starts = {eid: int(mat.columns.min()) for eid, mat in self.score_mats.items() if len(mat.columns)}
        episode_ends = {eid: int(mat.columns.max()) for eid, mat in self.score_mats.items() if len(mat.columns)}

        episode_id = next((eid for eid, s in episode_starts.items() if s <= start_frame <= episode_ends[eid]), None)
        if episode_id is None:
            after = [(s, eid) for eid, s in episode_starts.items() if s >= start_frame]
            if not after:
                return None, [], []
            episode_id = min(after)[1]

        score_mat = self.score_mats[episode_id]
        target_indices = self.synced_events[self.synced_events["frame_id"].between(start_frame, end_frame)].index
        event_rows = [i for i in score_mat.index if i in target_indices]
        frame_cols = [f for f in score_mat.columns if start_frame <= f <= end_frame]
        return episode_id, event_rows, frame_cols

    def get_matrix_slices(self, start_frame: int, end_frame: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return score_mat and dp_mat slices for the episode covering ``start_frame``.

        All-zero score columns are dropped from both slices for compactness.

        Parameters
        ----------
        start_frame, end_frame:
            Inclusive frame range.

        Returns
        -------
        score_mat, dp_mat:
            Slices for the selected episode. Empty DataFrames if no episode
            matches or the range has no overlap.
        """
        episode_id, event_rows, frame_cols = self._slice_episode(start_frame, end_frame)
        if episode_id is None or not event_rows or not frame_cols:
            return pd.DataFrame(), pd.DataFrame()

        score_mat = self.score_mats[episode_id]
        dp_mat = self.dp_mats[episode_id]
        frame_cols = [f for f in frame_cols if (score_mat[f] != 0).any()]

        if not frame_cols:
            return pd.DataFrame(), pd.DataFrame()
        return score_mat.loc[event_rows, frame_cols], dp_mat.loc[event_rows, frame_cols]

    def plot_features(self, start_frame: int, end_frame: int, ax: plt.Axes = None) -> plt.Axes:
        """Plot player_dist and ball_accel for a frame range.

        Parameters
        ----------
        events:
            Event data restricting plotted players to those who triggered events
            within the frame range. Candidate frames in ``self.cand_frames`` are
            always drawn as dashed vertical lines; when ``frame_id`` is present
            in ``events``, each event time is additionally marked with a black
            solid vertical line.
        start_frame, end_frame:
            Inclusive frame range to visualise.
        ax:
            Existing axes to draw on. A new figure is created when ``None``.

        Returns
        -------
        plt.Axes
        """
        plt.rcParams["font.size"] = 15

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        # tracking window
        mask = (self.tracking["frame_id"] >= start_frame) & (self.tracking["frame_id"] <= end_frame)
        window = self.tracking[mask].copy()

        ball = window[window["ball"]].set_index("frame_id").sort_index()
        players = window[window["player_id"].notna()].copy()

        # player_dist per player
        ball_xy = ball[["x", "y"]].rename(columns={"x": "ball_x", "y": "ball_y"})
        merged = players.join(ball_xy, on="frame_id", how="inner")
        merged["player_dist"] = np.sqrt((merged["x"] - merged["ball_x"]) ** 2 + (merged["y"] - merged["ball_y"]) ** 2)

        # Drop one-touch control duplicates so I/O markers don't overlap at the same frame
        target_events = self.synced_events[self.synced_events["frame_id"].between(start_frame, end_frame)]
        control_mask = target_events["spadl_type"] == "control"
        one_touch_mask = target_events["frame_id"].shift(-1) == target_events["frame_id"]
        target_events = target_events.loc[~(control_mask & one_touch_mask)]

        target_players = target_events["player_id"].unique().tolist()
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        player_colors: dict[str, str] = {}
        for i, pid in enumerate(target_players):
            player_colors[pid] = color_cycle[i % len(color_cycle)]

        for pid in target_players:
            grp = merged[merged["player_id"] == pid].set_index("frame_id").sort_index()
            ax.plot(grp.index, grp["player_dist"], color=player_colors[pid], alpha=0.7, label=pid)

        # ball_accel (scaled by 1/5 to match player_dist range)
        ax.plot(ball.index, ball["accel_v"] / 5, color="darkgray", label="ball_accel")

        target_cands = self.cand_frames[
            (self.cand_frames["frame_id"].between(start_frame, end_frame))
            & (self.cand_frames["player_id"].isin(target_players))
        ]
        for _, row in target_cands.iterrows():
            ax.axvline(row["frame_id"], color=player_colors[row["player_id"]], linestyle="--", alpha=0.7)

        for _, row in target_events.dropna(subset=["frame_id"]).iterrows():
            fid = row["frame_id"]
            color = player_colors[row["player_id"]]
            ax.axvline(fid, color=color, linestyle="-")

            if row["spadl_type"] in set(config.PASS_LIKE_OPEN + config.SET_PIECE):
                letter = "O"
            elif row["spadl_type"] in set(config.INCOMING):
                letter = "I"
            else:
                letter = "M"
            ax.scatter([fid], [25], s=300, c=color, zorder=5, clip_on=False)
            ax.text(
                fid,
                24.9,
                letter,
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
                zorder=6,
                clip_on=False,
            )

        ax.set_xlim(start_frame, end_frame)
        ax.set_ylim(0, 25)
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Player-ball distance (m)")
        ax.legend(loc="upper right", fontsize=12)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        return ax

    def plot_score_matrix(
        self,
        start_frame: int,
        end_frame: int,
        ax: plt.Axes = None,
        decimals: int = 2,
        cell_size: float = 0.9,
        cmap: str = "Reds",
    ) -> plt.Axes:
        """Render the score matrix slice as a heatmap-style figure.

        Cells are shaded by score (assumed in [0, 1]) using the given colormap.
        No sentinel row/column, ellipsis, or move arrows are drawn. Column
        headers for candidate frames matched on the optimal alignment path are
        rendered in bold.

        Parameters
        ----------
        start_frame, end_frame:
            Inclusive frame range (uses the same episode selection as ``get_matrix_slices``).
        decimals:
            Number of decimal places for cell values.
        cell_size:
            Cell fill size within its unit square (``0 < cell_size <= 1``).
        cmap:
            Matplotlib colormap name; defaults to ``Reds``.

        Returns
        -------
        plt.Axes
        """
        score_slice, _ = self.get_matrix_slices(start_frame, end_frame)
        if score_slice.empty:
            raise ValueError(f"No score slice available for frames [{start_frame}, {end_frame}].")

        n_rows, n_cols = score_slice.shape
        episode_id, _, frame_cols_full = self._slice_episode(start_frame, end_frame)

        if ax is None:
            # Match plot_dp_table figure width: it has a sentinel column plus an
            # optional ellipsis column on top of the actual frame columns.
            full_dp = self.dp_mats[episode_id]
            col_offset = 1 if frame_cols_full[0] != full_dp.columns[1] else 0
            fig_w = max(6.0, 0.6 * (n_cols + 1 + col_offset + 3))
            fig_h = max(3.0, 0.42 * (n_rows + 2))
            _, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=0.4, vmax=1.0)
        pad = (1 - cell_size) / 2

        for r in range(n_rows):
            for c in range(n_cols):
                score = float(score_slice.iat[r, c])
                ax.add_patch(
                    Rectangle(
                        (c + pad, -r - 1 + pad),
                        cell_size,
                        cell_size,
                        facecolor=cmap_obj(norm(score)),
                        edgecolor="black",
                        lw=0.5,
                        zorder=1,
                    )
                )
                text_color = "white" if score > 0.7 else "black"
                ax.text(
                    c + 0.5,
                    -r - 0.5,
                    f"{score:.{decimals}f}".lstrip("0"),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=text_color,
                    zorder=3,
                )

        # Column headers
        for c in range(n_cols):
            frame_id = score_slice.columns[c]
            ax.text(
                c + 0.5,
                0.25,
                str(frame_id),
                ha="center",
                va="bottom",
                fontsize=13,
                rotation=45,
                clip_on=False,
            )

        # Row headers: "{player_abbrev} {spadl_type} ({event_idx})".
        for r in range(n_rows):
            event_idx = score_slice.index[r]
            row = self.synced_events.loc[event_idx]
            player_id = row.get("player_id", "")
            if isinstance(player_id, str) and "_" in player_id:
                team, num = player_id.split("_", 1)
                player_short = f"{team[0].upper()}{num}"
            else:
                player_short = str(player_id)
            ax.text(
                -0.2,
                -r - 0.5,
                f"{player_short} {row['spadl_type']} ({event_idx})",
                ha="right",
                va="center",
                fontsize=13,
                clip_on=False,
            )

        ax.set_aspect("equal")
        ax.set_xlim(-2.6, n_cols + 0.2)
        ax.set_ylim(-n_rows - 0.5, 1.5)
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=13)
        return ax

    def plot_dp_table(
        self,
        start_frame: int,
        end_frame: int,
        ax: plt.Axes = None,
        decimals: int = 2,
        cell_size: float = 0.9,
    ) -> plt.Axes:
        """Render the NW DP table slice.

        Parameters
        ----------
        start_frame, end_frame:
            Inclusive frame range. Episode selection follows ``get_matrix_slices``.
        decimals:
            Number of decimal places for cell values.
        cell_size:
            Cell fill size within its unit square (``0 < cell_size <= 1``).
            Smaller values leave more whitespace between cells.

        Returns
        -------
        plt.Axes
        """
        episode_id, event_rows, frame_cols = self._slice_episode(start_frame, end_frame)
        if episode_id is None or not event_rows or not frame_cols:
            raise ValueError(f"No DP slice available for frames [{start_frame}, {end_frame}].")

        # Drop candidate-frame columns that score zero against every event in the slice
        score_mat = self.score_mats[episode_id]
        frame_cols = [f for f in frame_cols if (score_mat.loc[event_rows, f] != 0).any()]
        if not frame_cols:
            raise ValueError(f"No DP slice available for frames [{start_frame}, {end_frame}].")

        dp_mat = self.dp_mats[episode_id]
        trace_mat = self.trace_mats[episode_id]
        dp_slice = dp_mat.loc[[-1] + event_rows, [-1] + frame_cols]
        trace_slice = trace_mat.loc[[-1] + event_rows, [-1] + frame_cols]
        n_rows, n_cols = dp_slice.shape

        # Walk the full-episode path forward to collect cells that the optimal
        # alignment passes through, then map to slice (row_pos, col_pos).
        on_path_cells = set()
        if episode_id in self.paths:
            slice_event_index = list(dp_slice.index)
            slice_frame_cols = list(dp_slice.columns)
            full_event_index = list(dp_mat.index)
            full_frame_cols = list(dp_mat.columns)
            i, j = 0, 0
            for move in self.paths[episode_id]["move"]:
                if move == "diag":
                    i += 1
                    j += 1
                elif move == "repeat":
                    i += 1
                elif move == "up":
                    i += 1
                elif move == "left":
                    j += 1
                if i >= len(full_event_index) or j >= len(full_frame_cols):
                    continue
                row_label = full_event_index[i]
                col_label = full_frame_cols[j]
                if row_label in slice_event_index and col_label in slice_frame_cols:
                    on_path_cells.add((slice_event_index.index(row_label), slice_frame_cols.index(col_label)))

        if ax is None:
            fig_w = max(7.0, 0.8 * (n_cols + 3))
            fig_h = max(4.0, 0.55 * (n_rows + 2))
            _, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Insert an ellipsis row/column when the slice doesn't start from the
        # episode's first event / first candidate frame. Non-sentinel cells are
        # visually shifted by +1 along the affected axis so the ellipsis sits
        # in its own empty strip between the sentinel and the first slice cell.
        full_dp = self.dp_mats[episode_id]
        col_offset = 1 if frame_cols[0] != full_dp.columns[1] else 0
        row_offset = 1 if event_rows[0] != full_dp.index[1] else 0
        total_cols_vis = n_cols + col_offset
        total_rows_vis = n_rows + row_offset

        # Cell backgrounds + values (text in lower-right so move shapes can
        # occupy the upper/left region).
        pad = (1 - cell_size) / 2
        for r in range(n_rows):
            y = r if r == 0 else r + row_offset
            for c in range(n_cols):
                x = c if c == 0 else c + col_offset
                facecolor = "khaki" if (r, c) in on_path_cells else "white"
                ax.add_patch(
                    Rectangle(
                        (x + pad, -y - 1 + pad),
                        cell_size,
                        cell_size,
                        facecolor=facecolor,
                        edgecolor="black",
                        lw=0.5,
                        zorder=1,
                    )
                )
                ax.text(
                    x + 1 - pad - 0.05,
                    -y - 1 + pad + 0.05,
                    f"{dp_slice.iat[r, c]:.{decimals}f}",
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    zorder=3,
                )

        # Move indicator shapes: red = match, gray = gap.
        #   diag-match (0)  -> upper-left square
        #   down-gap (1)    -> gray top-half triangle pointing down
        #   right-gap (2)   -> gray left-half triangle pointing right
        #   down-match (3)  -> red top-half triangle pointing down
        move_color = {0: "tab:red", 1: "gray", 2: "gray", 3: "tab:red"}
        square_size = 0.45
        for r in range(n_rows):
            y = r if r == 0 else r + row_offset
            for c in range(n_cols):
                if r == 0 and c == 0:
                    continue
                if r == 0:
                    move = 2
                elif c == 0:
                    move = 1
                else:
                    move = int(trace_slice.iat[r, c])
                x = c if c == 0 else c + col_offset
                color = move_color[move]
                if move == 0:
                    ax.add_patch(
                        Rectangle(
                            (x + pad, -y - pad - square_size),
                            square_size,
                            square_size,
                            facecolor=color,
                            edgecolor="none",
                            zorder=2,
                        )
                    )
                elif move in (1, 3):
                    verts = [(x + pad, -y - pad), (x + 1 - pad, -y - pad), (x + 0.5, -y - 0.5)]
                    ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor="none", zorder=2))
                elif move == 2:
                    verts = [(x + pad, -y - pad), (x + pad, -y - 1 + pad), (x + 0.5, -y - 0.5)]
                    ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor="none", zorder=2))

        # Ellipsis markers in the empty shifted-out strip (cell area only).
        if col_offset:
            for r in range(n_rows):
                y = r if r == 0 else r + row_offset
                ax.text(1.5, -y - 0.5, "⋯", ha="center", va="center", fontsize=16, zorder=3)
        if row_offset:
            for c in range(n_cols):
                x = c if c == 0 else c + col_offset
                ax.text(x + 0.5, -1.5, "⋮", ha="center", va="center", fontsize=16, zorder=3)
        if col_offset and row_offset:
            ax.text(1.5, -1.5, "⋱", ha="center", va="center", fontsize=16, zorder=3)

        # Column headers (frame_id).
        matched_frames = set()
        if episode_id in self.paths:
            path = self.paths[episode_id]
            matched_frames = set(path.loc[path["move"].isin(["diag", "repeat"]), "frame_id"].dropna().astype(int))
        for c in range(1, n_cols):
            frame_id = dp_slice.columns[c]
            ax.text(
                c + col_offset + 0.5,
                0.25,
                str(frame_id),
                ha="center",
                va="bottom",
                color="tab:red" if frame_id in matched_frames else "k",
                fontsize=13,
                rotation=45,
                clip_on=False,
                fontweight="bold" if frame_id in matched_frames else "normal",
            )

        # Row headers: "{player_abbrev} {spadl_type} ({event_idx})".
        for r in range(1, n_rows):
            event_idx = dp_slice.index[r]
            row = self.synced_events.loc[event_idx]
            player_id = row.get("player_id", "")
            if isinstance(player_id, str) and "_" in player_id:
                team, num = player_id.split("_", 1)
                player_short = f"{team[0].upper()}{num}"
            else:
                player_short = str(player_id)
            ax.text(
                -0.2,
                -(r + row_offset) - 0.5,
                f"{player_short} {row['spadl_type']} ({event_idx})",
                ha="right",
                va="center",
                fontsize=13,
                clip_on=False,
            )

        # Single-row legend below the matrix showing each move's shape + color.
        legend_items = [
            ("square", "tab:red", "diag-match"),
            ("right_tri", "gray", "right-gap"),
            ("down_tri", "gray", "down-gap"),
            ("down_tri", "tab:red", "down-match"),
            ("square", "khaki", "optimal path"),
        ]
        legend_y = -total_rows_vis - 1.4
        shape_size = 0.6
        legend_x_start = -3.5  # figure's left edge (= xlim left)
        item_step = 3.3  # data units per legend item (shape + label + spacing)
        # Shift shorter labels rightward to balance perceived spacing.
        extra_offset = {1: 0.2, 2: -0.1, 3: -0.3}
        legend_fontsize = 11
        for i, (shape, color, label) in enumerate(legend_items):
            cx = legend_x_start + i * item_step + extra_offset.get(i, 0.0)
            if shape == "square":
                ax.add_patch(
                    Rectangle(
                        (cx, legend_y),
                        shape_size,
                        shape_size,
                        facecolor=color,
                        edgecolor="none",
                        zorder=3,
                        clip_on=False,
                    )
                )
            elif shape == "right_tri":
                verts = [
                    (cx, legend_y + shape_size),
                    (cx, legend_y),
                    (cx + shape_size, legend_y + shape_size / 2),
                ]
                ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor="none", zorder=3, clip_on=False))
            elif shape == "down_tri":
                verts = [
                    (cx, legend_y + shape_size),
                    (cx + shape_size, legend_y + shape_size),
                    (cx + shape_size / 2, legend_y),
                ]
                ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor="none", zorder=3, clip_on=False))
            ax.text(
                cx + shape_size + 0.2,
                legend_y + shape_size / 2,
                label,
                ha="left",
                va="center",
                fontsize=legend_fontsize,
                clip_on=False,
            )

        ax.set_aspect("equal")
        ax.set_xlim(-2.6, total_cols_vis + 0.2)
        ax.set_ylim(-total_rows_vis - 0.5, 1.5)
        ax.axis("off")
        return ax
