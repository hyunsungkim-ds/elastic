import argparse
import os
import sys
from datetime import timedelta
from typing import Dict

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, axes, collections, lines, text
from matplotlib.patches import Rectangle

import tools.matplotsoccer as mps
from tools.stats_perform_data import StatsPerformData, find_spadl_event_types

anim_config = {
    "sports": "soccer",  # soccer or basketball
    "figsize": (10.8, 7.2),
    "fontsize": 15,
    "player_size": 400,
    "ball_size": 150,
    "star_size": 150,
    "annot_size": 100,
    "cell_size": 50,
    "player_history": 20,
    "ball_history": 50,
}


class Animator:
    def __init__(
        self,
        trace_dict: Dict[str, pd.DataFrame] = None,
        bg_heatmaps: np.ndarray = None,
        player_sizes: np.ndarray = None,
        show_times=True,
        show_episodes=False,
        show_events=False,
        text_cols=None,  # column names for additional annotation
        rotate_pitch=False,
        anonymize=False,
        small_image=False,
        play_speed=1,
    ):
        self.trace_dict = trace_dict
        self.bg_heatmaps = bg_heatmaps
        self.sizes = player_sizes

        self.sports = anim_config["sports"]
        self.show_times = show_times
        self.show_episodes = show_episodes
        self.show_events = show_events
        self.text_cols = text_cols
        self.rotate_pitch = rotate_pitch
        self.anonymize = anonymize

        self.pitch_size = (105, 68) if self.sports == "soccer" else (30, 15)
        self.small_image = small_image
        self.play_speed = play_speed

        self.arg_dict = dict()

    @staticmethod
    def plot_players(
        traces: pd.DataFrame,
        ax: axes.Axes,
        sizes=750,
        alpha=1,
        anonymize=False,
    ):
        if len(traces.columns) == 0:
            return None

        color = "tab:red" if traces.columns[0].startswith("home_") else "tab:blue"
        x = traces[traces.columns[0::2]].values
        y = traces[traces.columns[1::2]].values
        size = sizes[0, 0] if isinstance(sizes, np.ndarray) else sizes
        scat = ax.scatter(x[0], y[0], s=size, c=color, alpha=alpha, zorder=2)

        players = [c[:-2] for c in traces.columns[0::2]]
        player_dict = dict(zip(players, np.arange(len(players)) + 1))
        plots = dict()
        annots = dict()

        for p in players:
            (plots[p],) = ax.plot([], [], c=color, alpha=alpha, ls=":", zorder=0)

            player_id = player_dict[p] if anonymize else int(p.split("_")[-1])
            annots[p] = ax.annotate(
                player_id,
                xy=traces.loc[0, [f"{p}_x", f"{p}_y"]],
                ha="center",
                va="center",
                color="w",
                fontsize=anim_config["fontsize"] - 2,
                fontweight="bold",
                annotation_clip=False,
                zorder=3,
            )
            annots[p].set_animated(True)

        return traces, sizes, scat, plots, annots

    @staticmethod
    def animate_players(
        t: int,
        inplay_records: pd.DataFrame,
        traces: pd.DataFrame,
        sizes: np.ndarray,
        scat: collections.PatchCollection,
        plots: Dict[str, lines.Line2D],
        annots: Dict[str, text.Annotation],
    ):
        x = traces[traces.columns[0::2]].values
        y = traces[traces.columns[1::2]].values
        scat.set_offsets(np.stack([x[t], y[t]]).T)

        if isinstance(sizes, np.ndarray):
            scat.set_sizes(sizes[t])

        for p in plots.keys():
            inplay_start = inplay_records.at[p, "start_index"]
            inplay_end = inplay_records.at[p, "end_index"]

            if t >= inplay_start:
                if t <= inplay_end:
                    t_from = max(t - anim_config["player_history"] + 1, inplay_start)
                    plots[p].set_data(traces.loc[t_from:t, f"{p}_x"], traces.loc[t_from:t, f"{p}_y"])
                    annots[p].set_position(traces.loc[t, [f"{p}_x", f"{p}_y"]].values)
                elif t == inplay_end + 1:
                    plots[p].set_alpha(0)
                    annots[p].set_alpha(0)

    @staticmethod
    def plot_ball(xy: pd.DataFrame, ax=axes.Axes, color="w", edgecolor="k", marker="o", show_path=True):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        scat = ax.scatter(
            x,
            y,
            s=anim_config["ball_size"],
            c=color,
            edgecolors=edgecolor,
            marker=marker,
            zorder=4,
        )

        if show_path:
            pathcolor = "k" if color in ["w", "darkorange"] else color
            (plot,) = ax.plot([], [], pathcolor, zorder=3)
        else:
            plot = None

        return x, y, scat, plot

    @staticmethod
    def animate_ball(
        t: int,
        x: np.ndarray,
        y: np.ndarray,
        scat: collections.PatchCollection,
        plot: lines.Line2D = None,
    ):
        scat.set_offsets(np.array([x[t], y[t]]))

        if plot is not None:
            t_from = max(t - anim_config["ball_history"], 0)
            plot.set_data(x[t_from : t + 1], y[t_from : t + 1])

    @staticmethod
    def plot_events(xy: pd.DataFrame, ax=axes.Axes, color="orange", edgecolor="k", marker="*"):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        scat = ax.scatter(
            x[0],
            y[0],
            s=anim_config["star_size"] if marker == "*" else anim_config["annot_size"],
            c=color,
            edgecolors=edgecolor if marker != "x" else None,
            marker=marker,
            zorder=100,
        )
        return x, y, scat

    @staticmethod
    def animate_events(t: int, x: np.ndarray, y: np.ndarray, scat: collections.PatchCollection):
        scat.set_offsets(np.array([x[t], y[t]]))

    def plot_init(self, ax: axes.Axes, trace_key: str):
        traces = self.trace_dict[trace_key].iloc[:: self.play_speed].copy()
        traces = traces.dropna(axis=1, how="all").reset_index(drop=True)
        xy_cols = [c for c in traces.columns if c.endswith("_x") or c.endswith("_y")]

        if self.rotate_pitch:
            traces[xy_cols[0::2]] = self.pitch_size[0] - traces[xy_cols[0::2]]
            traces[xy_cols[1::2]] = self.pitch_size[1] - traces[xy_cols[1::2]]

        inplay_records = []
        for c in xy_cols[::2]:
            inplay_index = traces[traces[c].notna()].index
            inplay_records.append([c[:-2], inplay_index[0], inplay_index[-1]])
        inplay_records = pd.DataFrame(inplay_records, columns=["object", "start_index", "end_index"])

        home_traces = traces[[c for c in xy_cols if c.startswith("home_")]].fillna(-100)
        away_traces = traces[[c for c in xy_cols if c.startswith("away_")]].fillna(-100)

        if trace_key == "main" and self.sizes is not None:
            if self.sizes.shape[1] == 2:  # team_poss
                sizes = self.sizes.fillna(0.5).values[(self.play_speed - 1) :: self.play_speed]
                home_sizes = np.repeat(sizes[:, [0]] * 500 + 500, home_traces.shape[1], axis=1)
                away_sizes = np.repeat(sizes[:, [1]] * 500 + 500, away_traces.shape[1], axis=1)
            else:  # player_poss
                n_players = home_traces.shape[1] // 2
                sizes = self.sizes.dropna(axis=1, how="all")
                sizes = sizes.fillna(1 / sizes.shape[1]).values[(self.play_speed - 1) :: self.play_speed]
                home_sizes = sizes[:, :n_players] * 1500 + 500
                away_sizes = sizes[:, n_players : n_players * 2] * 1500 + 500

        else:
            home_sizes = anim_config["player_size"]
            away_sizes = anim_config["player_size"]

        alpha = 1 if trace_key == "main" else 0.5
        home_args = self.plot_players(home_traces, ax, home_sizes, alpha, self.anonymize)
        away_args = self.plot_players(away_traces, ax, away_sizes, alpha, self.anonymize)

        ball_args = None
        if "ball_x" in traces.columns and traces["ball_x"].notna().any():
            ball_xy = traces[["ball_x", "ball_y"]]
            if trace_key == "main":
                if self.sports == "soccer":
                    ball_args = Animator.plot_ball(ball_xy, ax, "w", "k", "o")
                else:
                    ball_args = Animator.plot_ball(ball_xy, ax, "darkorange", "k", "o")
            else:
                ball_args = Animator.plot_ball(ball_xy, ax, trace_key, None, "*")

        self.trace_dict[trace_key] = traces
        self.arg_dict[trace_key] = {
            "inplay_records": inplay_records.set_index("object"),
            "home": home_args,
            "away": away_args,
            "ball": ball_args,
        }

    def run(self, cmap="jet", vmin=0, vmax=1, max_frames=np.inf, fps=10):
        if self.sports == "soccer":
            fig, ax = plt.subplots(figsize=anim_config["figsize"])
            mps.field("green", self.pitch_size[0], self.pitch_size[1], fig, ax, show=False)
            # fig.set_tight_layout(True)
        else:
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.set_xlim(-2, self.pitch_size[0] + 2)
            ax.set_ylim(-1, self.pitch_size[1] + 1)
            ax.axis("off")
            ax.grid(False)
            court = plt.imread("images/bball_court.png")
            ax.imshow(court, zorder=0, extent=[0, self.pitch_size[0], self.pitch_size[1], 0])

        for key in self.trace_dict.keys():
            self.plot_init(ax, key)

        traces = self.trace_dict["main"]
        text_y = self.pitch_size[1] + 1

        if self.bg_heatmaps is not None:
            hm_extent = (0, self.pitch_size[0], 0, self.pitch_size[1])
            hm = ax.imshow(self.bg_heatmaps[0], extent=hm_extent, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7)

        if self.show_times:
            timestamps = traces["timestamp"] if self.sports == "soccer" else traces["time_left"]
            timestamps = timestamps.dt.total_seconds() if isinstance(timestamps.iloc[0], timedelta) else timestamps
            timestamps_str = timestamps.apply(lambda x: f"{int(x // 60):02d}:{x % 60:05.2f}").values
            time_text = ax.text(
                0,
                text_y,
                timestamps_str[0],
                fontsize=anim_config["fontsize"],
                ha="left",
                va="bottom",
            )
            time_text.set_animated(True)

        if self.show_episodes:
            episodes_str = traces["episode"].apply(lambda x: f"Episode {x}")
            episodes_str = np.where(episodes_str == "Episode 0", "", episodes_str)
            text_x = self.pitch_size[0]
            episode_text = ax.text(
                text_x,
                text_y,
                episodes_str[0],
                fontsize=anim_config["fontsize"],
                ha="right",
                va="bottom",
            )
            episode_text.set_animated(True)

        if self.show_events:
            assert "event_type" in traces.columns
            events_str = traces.apply(lambda x: f"{x['event_type']} by {x['player_id']}", axis=1)
            events_str = np.where(events_str == "nan by nan", "", events_str)

            text_x = self.pitch_size[0] / 2
            event_text = ax.text(
                text_x,
                text_y,
                str(events_str[0]),
                fontsize=anim_config["fontsize"],
                ha="center",
                va="bottom",
            )
            event_text.set_animated(True)

            if "event_x" in traces.columns:
                event_args = Animator.plot_events(traces[["event_x", "event_y"]], ax, color="orange", marker="*")

            if "annot_x" in traces.columns:
                annot_args = Animator.plot_events(traces[["annot_x", "annot_y"]], ax, color="k", marker="X")

        if self.text_cols is not None:
            str_dict = {}
            text_dict = {}
            for i, col in enumerate(self.text_cols):
                str_dict[col] = f"{col}: " + np.where(traces[col].isna(), "", traces[col].astype(str))
                text_x = self.pitch_size[0] * i / 2
                text_dict[col] = ax.text(
                    text_x,
                    -1,
                    str(str_dict[col][0]),
                    fontsize=anim_config["fontsize"],
                    ha="left",
                    va="top",
                )
                text_dict[col].set_animated(True)

        def animate(t):
            for key in self.trace_dict.keys():
                inplay_records = self.arg_dict[key]["inplay_records"]
                home_args = self.arg_dict[key]["home"]
                away_args = self.arg_dict[key]["away"]
                ball_args = self.arg_dict[key]["ball"]

                if home_args is not None:
                    Animator.animate_players(t, inplay_records, *home_args)
                if away_args is not None:
                    Animator.animate_players(t, inplay_records, *away_args)
                if ball_args is not None:
                    Animator.animate_ball(t, *ball_args)

            if self.bg_heatmaps is not None:
                hm.set_array(self.bg_heatmaps[t])

            if self.show_times:
                time_text.set_text(str(timestamps_str[t]))

            if self.show_episodes:
                episode_text.set_text(str(episodes_str[t]))

            if self.show_events:
                event_text.set_text(events_str[t])

                if "event_x" in traces.columns:
                    Animator.animate_events(t, *event_args)

                if "annot_x" in traces.columns:
                    Animator.animate_events(t, *annot_args)

            if self.text_cols is not None:
                for col in self.text_cols:
                    text_dict[col].set_text(str(str_dict[col][t]))

        frames = min(max_frames, traces.shape[0])
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / fps)
        plt.close(fig)

        return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_index", type=int, default=None)
    parser.add_argument("--match_id", type=str, default=None)
    parser.add_argument("--load_preprocessed", action="store_true", default=False)
    parser.add_argument("--show_events", action="store_true", default=False)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=0)
    parser.add_argument("--segment_size", type=int, default=7500)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--step_size", type=int, default=1)

    args = parser.parse_known_args()[0]
    trace_files = np.sort([f for f in os.listdir("data/ajax/tracking") if f.endswith(".parquet")])

    if args.file_index is None:
        assert args.match_id is not None
        match_id = args.match_id
        file_index = np.where(trace_files == f"{match_id}.parquet")[0][0]

    elif args.match_id is None:
        assert args.file_index is not None
        file_index = args.file_index
        match_id = trace_files[file_index].split(".")[0]

    if args.load_preprocessed:
        match_events = pd.read_csv(f"data/ajax/event_processed/{match_id}.csv", header=0)
        tracking = pd.read_parquet(f"data/ajax/tracking_processed/{match_id}.parquet")

        print("1. Load the preprocessed event data and merge it with the tracking data")
        receive_events = []

        for i in match_events.index:
            event = match_events.loc[i]
            receive_frame = event["receive_frame_id"]
            next_frame = np.inf if i == match_events.index[-1] else match_events.at[i + 1, "frame"]

            if pd.notna(receive_frame) and receive_frame < next_frame:
                receive_events.append(
                    {
                        "period_id": event["period_id"],
                        "frame": receive_frame,
                        "utc_timestamp": None,
                        "synced_ts": event["receive_ts"],
                        "player_id": event["receiver_id"],
                        "spadl_type": "receive",
                    }
                )

        event_cols = ["utc_timestamp", "period_id", "frame", "player_id", "spadl_type"]
        aug_events = pd.concat([match_events, pd.DataFrame(receive_events)])[event_cols]

        aug_events = aug_events.dropna(subset="frame").sort_values(["frame", "utc_timestamp"], ignore_index=True)
        aug_events["frame"] = aug_events["frame"].astype(int)
        aug_events = aug_events[~aug_events.duplicated(subset="frame", keep="first")].drop("utc_timestamp", axis=1)
        aug_events["event_x"] = aug_events.apply(lambda e: tracking.at[e["frame"], "ball_x"], axis=1)
        aug_events["event_y"] = aug_events.apply(lambda e: tracking.at[e["frame"], "ball_y"], axis=1)
        aug_events.columns = ["period_id", "frame", "player_id", "event_type", "event_x", "event_y"]

        merged_df = pd.merge(aug_events, tracking.reset_index(), how="right").set_index("frame")
        merged_df[aug_events.columns[2:]] = merged_df[aug_events.columns[2:]].ffill()

    else:
        lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet")
        events = pd.read_parquet("data/ajax/event/event.parquet")
        events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"])
        events = events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True)

        print("1. Preprocess the event data and merge it with the tracking data")
        events = find_spadl_event_types(events)
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].set_index("player_id")
        match_events = events[(events["stats_perform_match_id"] == match_id) & (events["spadl_type"].notna())].copy()
        tracking = pd.read_parquet(f"data/ajax/tracking/{match_id}.parquet")

        match = StatsPerformData(match_lineup, match_events, tracking)
        match.refine_events()
        merged_df = StatsPerformData.merge_events_and_tracking(events, tracking, ffill=True)

    print("2. Animate selected trajectories")
    end_frame = merged_df.index[-1] if args.end_frame == 0 else args.end_frame
    break_frames = np.arange(args.start_frame, end_frame, args.segment_size)

    sampled_fps = round(args.fps / args.step_size, 1)
    if sampled_fps == int(sampled_fps):
        sampled_fps = int(sampled_fps)

    writer = animation.FFMpegWriter(fps=sampled_fps)
    os.makedirs("animations", exist_ok=True)

    for i, f_from in enumerate(break_frames):
        f_to = break_frames[i + 1] if i < len(break_frames) - 1 else end_frame
        print(f"Frames from {f_from} to {f_to}...")

        segment_df = merged_df.loc[f_from : f_to : args.step_size].copy()
        animator = Animator({"main": segment_df}, show_times=True, show_events=args.show_events)
        anim = animator.run()

        anim_path = f"animations/{file_index:03d}_{f_from}-{f_to}_fps{sampled_fps}.mp4"
        anim.save(anim_path, writer=writer)
