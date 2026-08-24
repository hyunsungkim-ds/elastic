import os
import sys
from typing import Callable, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from sync import config, utils
from sync.elastic_nw import ELASTIC_NW


class ELASTIC_Greedy(ELASTIC_NW):
    """Synchronize event and tracking data using greedy matching.

    Shares candidate frame selection and event-candidate scoring with ELASTIC_NW,
    but replaces the global Needleman-Wunsch alignment with greedy matching:
    each event in chronological order takes the best candidate frame within a
    qualifying window around its annotated timestamp, and termination events
    (goal, out, or control) are then detected between the matched frames of
    their surrounding events.

    Parameters
    ----------
    events: pd.DataFrame
        Event data to synchronize, according to schema sync.schema.elastic_event_schema.
    tracking: pd.DataFrame
        Tracking data to synchronize, according to schema sync.schema.tracking_schema.
    fps: float
        Tracking data FPS.
    """

    MAJOR_TYPES = config.PASS_LIKE_OPEN + config.SET_PIECE + config.INCOMING
    MINOR_TYPES = ["tackle", "dispossessed", "take_on"]

    def _annotated_frames(self, events: pd.DataFrame) -> pd.Series:
        """Map each event's annotated utc_timestamp to the nearest tracking frame."""
        annot_frames = pd.Series(np.nan, index=events.index)

        for period_id in events["period_id"].dropna().unique():
            period_events = events[events["period_id"] == period_id]
            period_frames = self.frames[self.frames["period_id"] == period_id].sort_values("utc_timestamp")
            if period_frames.empty:
                continue

            aligned = pd.merge_asof(
                period_events.sort_values("utc_timestamp").reset_index()[["index", "utc_timestamp"]],
                period_frames.reset_index()[["utc_timestamp", "frame_id"]],
                on="utc_timestamp",
                direction="nearest",
            )
            annot_frames.loc[aligned["index"]] = aligned["frame_id"].values

        return annot_frames

    @staticmethod
    def _find_window_time(event_type: str) -> float:
        if event_type in config.SET_PIECE:
            return config.TIME_SET_PIECE
        elif event_type in config.INCOMING:
            return config.TIME_INCOMING
        elif event_type in ELASTIC_Greedy.MINOR_TYPES:
            return config.TIME_MINOR
        else:  # PASS_LIKE_OPEN including bad_touch
            return config.TIME_PASS_LIKE_OPEN

    @staticmethod
    def _find_score_fn(event_type: str) -> Callable:
        if event_type in ["tackle", "dispossessed"]:
            return utils.nw_score_minor
        elif event_type == "take_on":
            return utils.nw_score_takeon
        else:
            return utils.nw_score_major

    @staticmethod
    def _best_candidate(window: pd.DataFrame, player_id: str, event_type: str) -> Tuple[float, float]:
        """Find the matching candidate frame for the given event in the window.

        Set pieces take the executing player's first candidate frame
        (the first player-ball distance minimum below 3 m among in-play frames);
        the other events take the highest-scoring candidate frame.
        """
        cands = window[window["player_id"] == player_id]
        if cands.empty:
            return np.nan, np.nan

        if event_type in config.SET_PIECE:
            return float(cands.iloc[0]["frame_id"]), np.nan

        # "control" was historically in config.INCOMING; keep is_incoming=True as in ELASTIC_NW.
        is_incoming = event_type in config.INCOMING + ["bad_touch", "tackle", "control"]
        scores = ELASTIC_Greedy._find_score_fn(event_type)(cands, player_id, incoming=is_incoming)

        best = int(np.argmax(scores))
        if scores[best] <= 0:
            return np.nan, np.nan
        return float(cands.iloc[best]["frame_id"]), float(scores[best])

    def run(self, events: pd.DataFrame = None, simplify_one_touch: bool = True) -> pd.DataFrame:
        """Runs greedy matching between events and candidate frames across the full match."""
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

        cands = self.cand_frames.sort_values("frame_id").reset_index(drop=True)
        cand_frame_ids = cands["frame_id"].to_numpy()

        def window_between(start_frame: float, end_frame: float) -> pd.DataFrame:
            lo = np.searchsorted(cand_frame_ids, start_frame, side="left")
            hi = np.searchsorted(cand_frame_ids, end_frame, side="right")
            return cands.iloc[lo:hi]

        stage1_types = ELASTIC_Greedy.MAJOR_TYPES + ELASTIC_Greedy.MINOR_TYPES
        annot_frames = self._annotated_frames(events)
        matched_frames = pd.Series(np.nan, index=events.index)
        matched_scores = pd.Series(np.nan, index=events.index)

        # Stage 1: Greedily match events in chronological order.
        min_frame = 0.0
        last_period = None

        stage1_mask = events["spadl_type"].isin(stage1_types)
        for i in tqdm(events.index[stage1_mask], desc="Greedy matching"):
            period_id = events.at[i, "period_id"]
            if period_id != last_period:
                period_frames = self.frames[self.frames["period_id"] == period_id]
                min_frame = float(period_frames.index.min()) if not period_frames.empty else 0.0
                last_period = period_id

            annot_frame = annot_frames[i]
            if np.isnan(annot_frame):
                continue

            player_id = events.at[i, "player_id"]
            event_type = events.at[i, "spadl_type"]

            s = ELASTIC_Greedy._find_window_time(event_type)
            start_frame = max(min_frame, annot_frame - s * self.fps)
            end_frame = annot_frame + s * self.fps
            window = window_between(start_frame, end_frame)

            frame, score = ELASTIC_Greedy._best_candidate(window, player_id, event_type)
            if not np.isnan(frame):
                matched_frames[i] = frame
                matched_scores[i] = score
                min_frame = frame

        # Stage 2: Detect each termination event (goal, out, or control) between the synchronized frames.
        stage1_idxs = events.index[stage1_mask].to_numpy()

        for i in tqdm(events.index[events["spadl_type"].isin(config.EVENT_END)], desc="Detecting terminations"):
            player_id = events.at[i, "player_id"]
            event_type = events.at[i, "spadl_type"]

            prev_matched = matched_frames.loc[:i].dropna()
            if prev_matched.empty:
                continue
            start_frame = float(prev_matched.iloc[-1])

            next_idxs = stage1_idxs[stage1_idxs > i]
            next_matched = matched_frames[next_idxs].dropna()
            end_frame = float(next_matched.iloc[0]) if not next_matched.empty else np.inf
            window = window_between(start_frame, end_frame)

            frame, score = ELASTIC_Greedy._best_candidate(window, player_id, event_type)
            if not np.isnan(frame):
                matched_frames[i] = frame
                matched_scores[i] = score

        events["frame_id"] = matched_frames.round()
        events["score"] = matched_scores
        events.loc[events["score"] < 0.5, "frame_id"] = np.nan
        events["synced_ts"] = events["frame_id"].map(self.frames["timestamp"].to_dict())

        self.synced_events = events.copy()

        if simplify_one_touch:
            control_mask = events["spadl_type"] == "control"
            one_touch_mask = (events["frame_id"].shift(-1) == events["frame_id"]) | events["frame_id"].shift(-1).isna()
            events = events.loc[~(control_mask & one_touch_mask)].reset_index(drop=True)

        return events
