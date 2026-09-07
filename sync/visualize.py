"""Figures that explain the ELASTIC_NW alignment: the player-ball distances with candidate frames,
the pairwise score matrix, and the NW DP table (paper Fig. 2 and Fig. 3).

Input: an ELASTIC_NW instance after run() and an inclusive frame range.
Output: a matplotlib Axes, optionally saved to save_path.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

from sync import config
from sync.elastic_nw import ELASTIC_NW


def _slice_episode(syncer: ELASTIC_NW, start_frame: int, end_frame: int) -> tuple[int, list, list]:
    """Pick the episode covering ``start_frame`` and return (episode_id, event_rows, frame_cols).

    Episode selection: the one whose frame range includes ``start_frame``;
    otherwise the next episode starting at or after ``start_frame``. Returns
    ``(None, [], [])`` if no episode matches. ``frame_cols`` is NOT filtered
    by zero scores — callers can drop them as needed.
    """
    assert syncer.synced_events is not None, "Call run() first to populate matrices."

    episode_starts = {eid: int(mat.columns.min()) for eid, mat in syncer.score_mats.items() if len(mat.columns)}
    episode_ends = {eid: int(mat.columns.max()) for eid, mat in syncer.score_mats.items() if len(mat.columns)}

    episode_id = next((eid for eid, s in episode_starts.items() if s <= start_frame <= episode_ends[eid]), None)
    if episode_id is None:
        after = [(s, eid) for eid, s in episode_starts.items() if s >= start_frame]
        if not after:
            return None, [], []
        episode_id = min(after)[1]

    score_mat = syncer.score_mats[episode_id]
    target_indices = syncer.synced_events[syncer.synced_events["frame_id"].between(start_frame, end_frame)].index
    event_rows = [i for i in score_mat.index if i in target_indices]
    frame_cols = [f for f in score_mat.columns if start_frame <= f <= end_frame]
    return episode_id, event_rows, frame_cols


def get_matrix_slices(syncer: ELASTIC_NW, start_frame: int, end_frame: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return score_mat and dp_mat slices for the episode covering ``start_frame``.

    All-zero score columns are dropped from both slices for compactness.

    Parameters
    ----------
    syncer:
        Synchronizer after ``run()``.
    start_frame, end_frame:
        Inclusive frame range.

    Returns
    -------
    score_mat, dp_mat:
        Slices for the selected episode. Empty DataFrames if no episode matches or the range has no overlap.
    """
    episode_id, event_rows, frame_cols = _slice_episode(syncer, start_frame, end_frame)
    if episode_id is None or not event_rows or not frame_cols:
        return pd.DataFrame(), pd.DataFrame()

    score_mat = syncer.score_mats[episode_id]
    dp_mat = syncer.dp_mats[episode_id]
    frame_cols = [f for f in frame_cols if (score_mat[f] != 0).any()]

    if not frame_cols:
        return pd.DataFrame(), pd.DataFrame()
    return score_mat.loc[event_rows, frame_cols], dp_mat.loc[event_rows, frame_cols]


def plot_features(
    syncer: ELASTIC_NW,
    start_frame: int,
    end_frame: int,
    highlight: bool = True,
    ax: plt.Axes = None,
    save_path: str = None,
) -> plt.Axes:
    """Plot player_dist and ball_accel for a frame range.

    Parameters
    ----------
    syncer:
        Synchronizer after ``run()``; its candidate frames are drawn as black dashed lines.
    start_frame, end_frame:
        Inclusive frame range to visualise.
    highlight_selected:
        Mark the candidate frames selected by the alignment with a red solid line and a
        P/C/M label. When ``False``, every candidate frame stays a plain dashed line.
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
    mask = (syncer.tracking["frame_id"] >= start_frame) & (syncer.tracking["frame_id"] <= end_frame)
    window = syncer.tracking[mask].copy()

    ball = window[window["ball"]].set_index("frame_id").sort_index()
    players = window[window["player_id"].notna()].copy()

    # player_dist per player
    ball_xy = ball[["x", "y"]].rename(columns={"x": "ball_x", "y": "ball_y"})
    merged = players.join(ball_xy, on="frame_id", how="inner")
    merged["player_dist"] = np.sqrt((merged["x"] - merged["ball_x"]) ** 2 + (merged["y"] - merged["ball_y"]) ** 2)

    # Drop one-touch control duplicates so I/O markers don't overlap at the same frame
    target_events = syncer.synced_events[syncer.synced_events["frame_id"].between(start_frame, end_frame)]
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
    # ax.plot(ball.index, ball["accel_v"] / 5, color="darkgray", label="ball_accel")

    target_cands = syncer.cand_frames[
        (syncer.cand_frames["frame_id"].between(start_frame, end_frame))
        & (syncer.cand_frames["player_id"].isin(target_players))
    ]
    for _, row in target_cands.iterrows():
        # ax.axvline(row["frame_id"], color=player_colors[row["player_id"]], linestyle="--", alpha=0.7)
        ax.axvline(row["frame_id"], color="k", linestyle="--", alpha=0.7)

    if highlight:
        for _, row in target_events.dropna(subset=["frame_id"]).iterrows():
            fid = row["frame_id"]
            ax.axvline(fid, color="tab:red", linestyle="-")

            if row["spadl_type"] in set(config.PASS_LIKE_OPEN + config.SET_PIECE):
                letter = "P"
            elif row["spadl_type"] in set(config.INCOMING + ["control"]):
                letter = "C"
            else:
                letter = "M"
            ax.scatter([fid], [25], s=300, c="tab:red", marker="s", zorder=5, clip_on=False)
            ax.text(
                fid,
                24.9,
                letter,
                ha="center",
                va="center",
                color="white",
                fontsize=14,
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
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    return ax


def plot_score_matrix(
    syncer: ELASTIC_NW,
    start_frame: int,
    end_frame: int,
    ax: plt.Axes = None,
    decimals: int = 2,
    cell_size: float = 0.9,
    cmap: str = "Reds",
    save_path: str = None,
) -> plt.Axes:
    """Render the score matrix slice as a heatmap-style figure.

    Cells are shaded by score (assumed in [0, 1]) using the given colormap.
    No sentinel row/column, ellipsis, or move arrows are drawn. Column
    headers for candidate frames matched on the optimal alignment path are
    rendered in bold.

    Parameters
    ----------
    syncer:
        Synchronizer after ``run()``.
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
    score_slice, _ = get_matrix_slices(syncer, start_frame, end_frame)
    if score_slice.empty:
        raise ValueError(f"No score slice available for frames [{start_frame}, {end_frame}].")

    n_rows, n_cols = score_slice.shape
    episode_id, _, frame_cols_full = _slice_episode(syncer, start_frame, end_frame)

    if ax is None:
        # Match plot_dp_table figure width: it has a sentinel column plus an
        # optional ellipsis column on top of the actual frame columns.
        full_dp = syncer.dp_mats[episode_id]
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
        row = syncer.synced_events.loc[event_idx]
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
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    return ax


def plot_dp_table(
    syncer: ELASTIC_NW,
    start_frame: int,
    end_frame: int,
    ax: plt.Axes = None,
    decimals: int = 2,
    cell_size: float = 0.9,
    save_path: str = None,
) -> plt.Axes:
    """Render the NW DP table slice.

    Parameters
    ----------
    syncer:
        Synchronizer after ``run()``.
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
    episode_id, event_rows, frame_cols = _slice_episode(syncer, start_frame, end_frame)
    if episode_id is None or not event_rows or not frame_cols:
        raise ValueError(f"No DP slice available for frames [{start_frame}, {end_frame}].")

    # Drop candidate-frame columns that score zero against every event in the slice
    score_mat = syncer.score_mats[episode_id]
    frame_cols = [f for f in frame_cols if (score_mat.loc[event_rows, f] != 0).any()]
    if not frame_cols:
        raise ValueError(f"No DP slice available for frames [{start_frame}, {end_frame}].")

    dp_mat = syncer.dp_mats[episode_id]
    trace_mat = syncer.trace_mats[episode_id]
    dp_slice = dp_mat.loc[[-1] + event_rows, [-1] + frame_cols]
    trace_slice = trace_mat.loc[[-1] + event_rows, [-1] + frame_cols]
    n_rows, n_cols = dp_slice.shape

    # Walk the full-episode path forward to collect cells that the optimal
    # alignment passes through, then map to slice (row_pos, col_pos).
    on_path_cells = set()
    if episode_id in syncer.paths:
        slice_event_index = list(dp_slice.index)
        slice_frame_cols = list(dp_slice.columns)
        full_event_index = list(dp_mat.index)
        full_frame_cols = list(dp_mat.columns)
        i, j = 0, 0
        for move in syncer.paths[episode_id]["move"]:
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
    full_dp = syncer.dp_mats[episode_id]
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
    if episode_id in syncer.paths:
        path = syncer.paths[episode_id]
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
        row = syncer.synced_events.loc[event_idx]
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
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    return ax
