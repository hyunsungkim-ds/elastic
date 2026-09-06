"""SwiftEvent-based synchronizer of Biermann et al. (2023) for benchmarking.

Reimplements a two-class multivariate Gaussian classifier over feature windows,
applied as an informed MAP refinement around each annotated event timestamp.
The classifier is supervised, so it must be fitted on matches with GT timestamps
before synchronizing a held-out match (leave-one-match-out in the benchmark).
"""

import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.polynomial import polynomial as npoly

from sync import config, schema, utils

# ---- SwiftEvent FR3* hyperparameters (Biermann et al., 2023) ----
WIN_LEN = 25  # window length in frames (1 s at 25 FPS)
N_POLY_WEIGHTS = 6  # distance-window polynomial weights (fit of degree 5)
SEARCH_QUANTILE = 0.005  # two-sided error quantile defining the search interval
OUTLIER_THRESHOLD = 0.0  # keep the annotated frame if max probability <= tau
COV_RIDGE = 1e-8  # relative ridge on covariance diagonals for stable inversion

HALF_WIN = WIN_LEN // 2
OUTGOING_TYPES = config.PASS_LIKE_OPEN + config.SET_PIECE


# ---- Signals and window features ----


def _match_signals(tracking: pd.DataFrame) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """Per-frame player-ball distances and ball acceleration from long-format tracking."""
    ball = tracking.loc[tracking["ball"], ["frame_id", "x", "y"]].drop_duplicates("frame_id").sort_values("frame_id")
    frames = ball["frame_id"].to_numpy()
    bxy = ball[["x", "y"]].to_numpy()

    players = tracking.loc[~tracking["ball"], ["frame_id", "player_id", "x", "y"]]
    px = players.pivot(index="frame_id", columns="player_id", values="x").reindex(frames)
    py = players.pivot(index="frame_id", columns="player_id", values="y").reindex(frames)
    dist = np.sqrt((px.to_numpy() - bxy[:, [0]]) ** 2 + (py.to_numpy() - bxy[:, [1]]) ** 2)

    # Second difference of the ball position within each contiguous in-play run,
    # edge-filled so the run boundaries do not shrink the window coverage.
    acc = np.full(len(frames), np.nan)
    starts = np.flatnonzero(np.diff(frames, prepend=frames[0] - 2) != 1)
    for s, e in zip(starts, np.append(starts[1:], len(frames))):
        if e - s < 3:
            continue
        acc[s + 1 : e - 1] = np.linalg.norm(bxy[s + 2 : e] - 2 * bxy[s + 1 : e - 1] + bxy[s : e - 2], axis=1)
        acc[s] = acc[s + 1]
        acc[e - 1] = acc[e - 2]

    return frames, list(px.columns), dist, acc


def _valid_runs(frames: np.ndarray, values: np.ndarray) -> list[slice]:
    """Slices of consecutive frames with non-NaN values, long enough for a window."""
    idx = np.flatnonzero(~np.isnan(values))
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(frames[idx]) != 1)) + 1
    return [slice(c[0], c[-1] + 1) for c in np.split(idx, breaks) if len(c) >= WIN_LEN]


def _window_features(d_windows: np.ndarray, a_windows: np.ndarray) -> np.ndarray:
    """FR3 feature matrix for stacked distance/acceleration windows of shape (n, WIN_LEN)."""
    poly = npoly.polyfit(np.arange(WIN_LEN), d_windows.T, N_POLY_WEIGHTS - 1).T
    d_curve = (d_windows[:, 2:] - d_windows[:, :-2]) / 2
    a_curve = (a_windows[:, 2:] - a_windows[:, :-2]) / 2
    return np.column_stack(
        (
            poly,
            d_windows.min(axis=1),
            d_windows.argmin(axis=1),
            d_curve.mean(axis=1),
            d_curve[:, HALF_WIN - 1],
            a_curve.mean(axis=1),
            a_windows.argmax(axis=1),
        )
    )


def annotated_frames(events: pd.DataFrame, tracking: pd.DataFrame) -> np.ndarray:
    """Nearest tracking frame_id to each event's utc_timestamp, per period."""
    frames = tracking[["frame_id", "period_id", "utc_timestamp"]].drop_duplicates("frame_id").sort_values("frame_id")
    out = np.full(len(events), -1, dtype=np.int64)
    for p, pf in frames.groupby("period_id"):
        mask = (events["period_id"] == p).to_numpy()
        if not mask.any():
            continue
        frame_utcs = pf["utc_timestamp"].to_numpy()
        event_utcs = events.loc[mask, "utc_timestamp"].to_numpy()
        pos = np.clip(np.searchsorted(frame_utcs, event_utcs), 1, len(frame_utcs) - 1)
        take_left = (event_utcs - frame_utcs[pos - 1]) <= (frame_utcs[pos] - event_utcs)
        out[mask] = pf["frame_id"].to_numpy()[pos - take_left]
    return out


# ---- Model ----


class BiermannModel:
    """Two-class Gaussian window classifier with a data-driven search interval."""

    @staticmethod
    def match_statistics(events: pd.DataFrame, gt_events: pd.DataFrame, tracking: pd.DataFrame) -> dict:
        """Per-match training statistics, computed once and reusable across CV folds."""
        assert len(events) == len(gt_events), "Input and GT events must correspond 1:1."
        frames, player_ids, dist, acc = _match_signals(tracking)

        out_mask = gt_events["spadl_type"].isin(OUTGOING_TYPES).to_numpy()
        gt_out = gt_events.loc[out_mask]
        pos_by_player = {p: g["frame_id"].to_numpy(dtype=np.int64) for p, g in gt_out.groupby("player_id")}

        n_neg, neg_sum, neg_sqsum, pos_rows = 0, 0.0, 0.0, []
        for j, player_id in enumerate(player_ids):
            for run in _valid_runs(frames, dist[:, j] + acc):
                feats = _window_features(
                    sliding_window_view(dist[run, j], WIN_LEN), sliding_window_view(acc[run], WIN_LEN)
                )
                centers = frames[run][HALF_WIN:-HALF_WIN]
                labels = np.isin(centers, pos_by_player.get(player_id, ()))
                neg = feats[~labels]
                n_neg += len(neg)
                neg_sum += neg.sum(axis=0)
                neg_sqsum += neg.T @ neg
                if labels.any():
                    pos_rows.append(feats[labels])

        annot = annotated_frames(events, tracking)
        return {
            "n_neg": n_neg,
            "neg_sum": neg_sum,
            "neg_sqsum": neg_sqsum,
            "pos_features": np.vstack(pos_rows),
            "annot_errors": annot[out_mask] - gt_out["frame_id"].to_numpy(),
        }

    def fit(self, stats: list[dict]) -> "BiermannModel":
        """Estimate class Gaussians and the search interval from per-match statistics."""
        pos = np.vstack([s["pos_features"] for s in stats])
        n_neg = sum(s["n_neg"] for s in stats)
        neg_mean = sum(s["neg_sum"] for s in stats) / n_neg
        neg_cov = (sum(s["neg_sqsum"] for s in stats) - n_neg * np.outer(neg_mean, neg_mean)) / (n_neg - 1)

        self.mean_ = {0: neg_mean, 1: pos.mean(axis=0)}
        self.prec_ = {0: _stable_inverse(neg_cov), 1: _stable_inverse(np.cov(pos.T))}

        errors = np.concatenate([s["annot_errors"] for s in stats])
        lo_q, hi_q = np.quantile(errors, [SEARCH_QUANTILE, 1 - SEARCH_QUANTILE])
        self.search_interval_ = (-int(np.ceil(hi_q)), -int(np.floor(lo_q)))
        return self

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        """Pass probability of Eq. (1): normalized class-1 density gated by Mahalanobis distances."""
        z1 = feats - self.mean_[1]
        z0 = feats - self.mean_[0]
        d1 = np.einsum("ij,jk,ik->i", z1, self.prec_[1], z1)
        d0 = np.einsum("ij,jk,ik->i", z0, self.prec_[0], z0)
        return np.where(d1 <= d0, np.exp(-0.5 * d1), 0.0)


def _stable_inverse(cov: np.ndarray) -> np.ndarray:
    return np.linalg.inv(cov + COV_RIDGE * np.diag(cov).mean() * np.eye(len(cov)))


# ---- Synchronizer ----


class BiermannSync:
    """Synchronize outgoing events by MAP refinement around annotated frames.

    Mirrors the ``run`` contract of ETSY/ELASTIC_*: returns the input events with
    ``frame_id`` and ``synced_ts`` filled. Incoming and minor events are not
    covered by the method and keep ``frame_id`` = NaN.

    Parameters
    ----------
    events : pd.DataFrame
        Event data to synchronize, according to schema.elastic_event_schema.
    tracking : pd.DataFrame
        Tracking data, according to schema.tracking_schema.
    model : BiermannModel
        Classifier fitted on other matches.
    """

    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, model: BiermannModel, fps: int = 25):
        schema.elastic_event_schema.validate(events)
        schema.tracking_schema.validate(tracking)

        assert list(events.index.unique()) == [i for i in range(len(events))]
        assert list(tracking.index.unique()) == [i for i in range(len(tracking))]

        self.events = events.copy()
        self.tracking = tracking
        self.model = model
        self.fps = fps

        time_cols = ["frame_id", "period_id", "timestamp", "utc_timestamp"]
        self.frames = self.tracking[time_cols].drop_duplicates().sort_values("frame_id").set_index("frame_id")
        self.frames["timestamp"] = self.frames["timestamp"].apply(utils.seconds_to_timestamp)

    def _refine_event(self, annot_frame: int, dist_col: np.ndarray, acc: np.ndarray, frames: np.ndarray) -> float:
        """Return the argmax-probability frame in the search interval, or the annotated frame."""
        lo, hi = self.model.search_interval_
        cand = np.arange(annot_frame + lo, annot_frame + hi + 1)
        pos = np.clip(np.searchsorted(frames, cand), HALF_WIN, len(frames) - HALF_WIN - 1)

        # A candidate is valid if its window lies fully within one in-play run
        # and the player's distance signal has no gaps inside it.
        valid = (frames[pos] == cand) & (frames[pos + HALF_WIN] - frames[pos - HALF_WIN] == WIN_LEN - 1)
        if not valid.any():
            return annot_frame
        offsets = np.arange(-HALF_WIN, HALF_WIN + 1)
        d_windows = dist_col[pos[valid, None] + offsets]
        a_windows = acc[pos[valid, None] + offsets]
        nan_free = ~(np.isnan(d_windows).any(axis=1) | np.isnan(a_windows).any(axis=1))
        if not nan_free.any():
            return annot_frame

        probas = self.model.predict_proba(_window_features(d_windows[nan_free], a_windows[nan_free]))
        if probas.max() <= OUTLIER_THRESHOLD:
            return annot_frame
        return cand[valid][nan_free][probas.argmax()]

    def run(self) -> pd.DataFrame:
        """Applies the MAP refinement and returns the synced events."""
        frames, player_ids, dist, acc = _match_signals(self.tracking)
        annot = annotated_frames(self.events, self.tracking)
        player_col = {p: j for j, p in enumerate(player_ids)}

        matched = np.full(len(self.events), np.nan)
        for i in self.events.index[self.events["spadl_type"].isin(OUTGOING_TYPES)]:
            j = player_col.get(self.events.at[i, "player_id"])
            if j is None:
                matched[i] = annot[i]
                continue
            matched[i] = self._refine_event(annot[i], dist[:, j], acc, frames)

        self.events["frame_id"] = matched
        self.events["synced_ts"] = self.events["frame_id"].map(self.frames["timestamp"].to_dict())
        return self.events
