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
from tools.preprocessor import Preprocessor, find_spadl_event_types

anim_config = {
    "sports": "soccer",  # soccer or basketball
    "figsize": (10.8, 7.2),
    "fontsize": 15,
    "player_size": 400,
    "ball_size": 150,
    "star_size": 150,
    "cell_size": 50,
    "player_history": 20,
    "ball_history": 50,
}


class Animator:
    def __init__(
        self,
        trace_dict: Dict[str, pd.DataFrame] = None,
        bg_heatmaps: np.ndarray = None,
        player_dests: pd.DataFrame = None,
        player_sizes: np.ndarray = None,
        show_times=True,
        show_episodes=False,
        show_events=False,
        annot_cols=None,  # column names for additional annotation
        rotate_pitch=False,
        anonymize=False,
        small_image=False,
        play_speed=1,
    ):
        self.trace_dict = trace_dict
        self.bg_heatmaps = bg_heatmaps
        self.dests = player_dests
        self.sizes = player_sizes

        self.sports = anim_config["sports"]
        self.show_times = show_times
        self.show_episodes = show_episodes
        self.show_events = show_events
        self.annot_cols = annot_cols
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
        # ls = "-" if alpha == 1 else "--"

        for p in players:
            (plots[p],) = ax.plot([], [], c=color, alpha=alpha, ls=":", zorder=0)

            player_num = player_dict[p] if anonymize else int(p.split("_")[-1])
            annots[p] = ax.annotate(
                player_num,
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
            s=anim_config["star_size"],
            c=color,
            edgecolors=edgecolor,
            marker=marker,
            zorder=3,
        )
        return x, y, scat

    @staticmethod
    def animate_events(t: int, x: np.ndarray, y: np.ndarray, scat: collections.PatchCollection):
        scat.set_offsets(np.array([x[t], y[t]]))

    @staticmethod
    def plot_intent(ax=axes.Axes, patch_size=3):
        patch = Rectangle(xy=(0, 0), width=patch_size, height=patch_size, color="lime", alpha=0, zorder=0)
        ax.add_patch(patch)
        return patch

    @staticmethod
    def animate_intent(t: int, intents: np.array, patch=Rectangle, patch_size=3):
        patch.set_xy([intents[t, 0] - patch_size / 2, intents[t, 1] - patch_size / 2])
        patch.set_alpha(0.5)

    def plot_guides(traces: pd.DataFrame, dests: pd.DataFrame, ax=axes.Axes):
        players = [c[:-2] for c in dests.columns if c.endswith("_x")]
        dest_x = dests[[c for c in dests.columns if c.endswith("_x")]].values
        dest_y = dests[[c for c in dests.columns if c.endswith("_y")]].values
        line_x = dict()
        line_y = dict()
        guide_scat = ax.scatter(dest_x[0], dest_y[0], s=anim_config["cell_size"], c="magenta", marker="s", zorder=3)
        guide_plots = dict()

        for p in players:
            # color = "tab:red" if p[0] == "H" else "tab:blue"
            line_x[p] = np.stack([traces[f"{p}_x"].values, dests[f"{p}_x"].values]).T
            line_y[p] = np.stack([traces[f"{p}_y"].values, dests[f"{p}_y"].values]).T
            (guide_plots[p],) = ax.plot(line_x[p][0], line_y[p][0], c="purple", lw=1.5, zorder=1)

        return dest_x, dest_y, line_x, line_y, guide_scat, guide_plots

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
        annot_y = self.pitch_size[1] + 1

        if self.bg_heatmaps is not None:
            hm_extent = (0, self.pitch_size[0], 0, self.pitch_size[1])
            hm = ax.imshow(self.bg_heatmaps[0], extent=hm_extent, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7)

        if self.dests is not None:
            dest_x, dest_y, line_x, line_y, guide_scat, guide_plots = Animator.plot_guides(traces, self.dests, ax)

        if self.show_times:
            timestamps = traces["timestamp"] if self.sports == "soccer" else traces["time_left"]
            timestamps = timestamps.dt.total_seconds() if isinstance(timestamps.iloc[0], timedelta) else timestamps
            time_texts = timestamps.apply(lambda x: f"{int(x // 60):02d}:{x % 60:05.2f}").values
            time_annot = ax.text(
                0,
                annot_y,
                time_texts[0],
                fontsize=anim_config["fontsize"],
                ha="left",
                va="bottom",
            )
            time_annot.set_animated(True)

        if self.show_episodes:
            episode_texts = traces["episode"].apply(lambda x: f"Episode {x}")
            episode_texts = np.where(episode_texts == "Episode 0", "", episode_texts)
            annot_x = self.pitch_size[0]
            episode_annot = ax.text(
                annot_x,
                annot_y,
                episode_texts[0],
                fontsize=anim_config["fontsize"],
                ha="right",
                va="bottom",
            )
            episode_annot.set_animated(True)

        if self.show_events:
            assert "event_type" in traces.columns
            event_texts = traces.apply(lambda x: f"{x['event_type']} by {x['event_player']}", axis=1)
            event_texts = np.where(event_texts == "nan by nan", "", event_texts)

            # elif "event_types" in traces.columns:
            #     event_texts = traces["event_types"].fillna(method="ffill")
            #     event_texts = np.where(event_texts.isna(), "", event_texts)

            annot_x = self.pitch_size[0] / 2
            event_annot = ax.text(
                annot_x,
                annot_y,
                str(event_texts[0]),
                fontsize=anim_config["fontsize"],
                ha="center",
                va="bottom",
            )
            event_annot.set_animated(True)

            if "event_x" in traces.columns:
                event_args = Animator.plot_events(traces[["event_x", "event_y"]], ax)

        if self.annot_cols is not None:
            text_dict = {}
            annot_dict = {}
            for i, col in enumerate(self.annot_cols):
                text_dict[col] = f"{col}: " + np.where(traces[col].isna(), "", traces[col].astype(str))
                annot_x = self.pitch_size[0] * i / 2
                annot_dict[col] = ax.text(
                    annot_x,
                    -1,
                    str(text_dict[col][0]),
                    fontsize=anim_config["fontsize"],
                    ha="left",
                    va="top",
                )
                annot_dict[col].set_animated(True)

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

            if self.dests is not None:
                guide_scat.set_offsets(np.stack([dest_x[t], dest_y[t]]).T)
                for p, plot in guide_plots.items():
                    plot.set_data(line_x[p][t], line_y[p][t])

            if self.show_times:
                time_annot.set_text(str(time_texts[t]))

            if self.show_episodes:
                episode_annot.set_text(str(episode_texts[t]))

            if self.show_events:
                event_annot.set_text(event_texts[t])
                if "event_x" in traces.columns:
                    Animator.animate_events(t, *event_args)

            if self.annot_cols is not None:
                for col in self.annot_cols:
                    annot_dict[col].set_text(str(text_dict[col][t]))

        frames = min(max_frames, traces.shape[0])
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / fps)
        plt.close(fig)

        return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_index", type=int, default=None)
    parser.add_argument("--game_id", type=str, default=None)
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
        assert args.game_id is not None
        game_id = args.game_id
        file_index = np.where(trace_files == f"{game_id}.parquet")[0][0]

    elif args.game_id is None:
        assert args.file_index is not None
        file_index = args.file_index
        game_id = trace_files[file_index].split(".")[0]

    if args.load_preprocessed:
        game_events = pd.read_csv(f"data/ajax/event_processed/{game_id}.csv", header=0)
        traces = pd.read_parquet(f"data/ajax/tracking_processed/{game_id}.parquet")

        print("1. Load the preprocessed event data and merge it with the tracking data")
        receive_events = []

        for i in game_events.index:
            event = game_events.loc[i]
            receive_frame = event["receive_frame"]
            next_frame = np.inf if i == game_events.index[-1] else game_events.at[i + 1, "frame"]

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
        aug_events = pd.concat([game_events, pd.DataFrame(receive_events)])[event_cols]

        aug_events = aug_events.dropna(subset="frame").sort_values(["frame", "utc_timestamp"], ignore_index=True)
        aug_events["frame"] = aug_events["frame"].astype(int)
        aug_events = aug_events[~aug_events.duplicated(subset="frame", keep="first")].drop("utc_timestamp", axis=1)
        aug_events["event_x"] = aug_events.apply(lambda e: traces.at[e["frame"], "ball_x"], axis=1)
        aug_events["event_y"] = aug_events.apply(lambda e: traces.at[e["frame"], "ball_y"], axis=1)
        aug_events.columns = ["period_id", "frame", "event_player", "event_type", "event_x", "event_y"]

        combined_traces = pd.merge(aug_events, traces.reset_index(), how="right").set_index("frame")
        combined_traces[aug_events.columns[2:]] = combined_traces[aug_events.columns[2:]].ffill()

    else:
        lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet")
        events = pd.read_parquet("data/ajax/event/event.parquet")
        events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"])
        events = events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True)

        print("1. Preprocess the event data and merge it with the tracking data")
        events = find_spadl_event_types(events)
        game_lineup = lineups.loc[lineups["stats_perform_match_id"] == game_id].set_index("player_id")
        game_events = events[(events["stats_perform_match_id"] == game_id) & (events["spadl_type"].notna())].copy()
        traces = pd.read_parquet(f"data/ajax/tracking/{game_id}.parquet")

        proc = Preprocessor(game_lineup, game_events, traces)
        proc.refine_events()
        combined_traces = proc.merge_events_and_traces(ffill=True)

    print("2. Animate selected trajectories")
    end_frame = combined_traces.index[-1] if args.end_frame == 0 else args.end_frame
    break_frames = np.arange(args.start_frame, end_frame, args.segment_size)

    sampled_fps = round(args.fps / args.step_size, 1)
    if sampled_fps == int(sampled_fps):
        sampled_fps = int(sampled_fps)

    writer = animation.FFMpegWriter(fps=sampled_fps)
    os.makedirs("animations", exist_ok=True)

    for i, f_from in enumerate(break_frames):
        f_to = break_frames[i + 1] if i < len(break_frames) - 1 else end_frame
        print(f"Frames from {f_from} to {f_to}...")

        segment_traces = combined_traces.loc[f_from : f_to : args.step_size].copy()
        animator = Animator({"main": segment_traces}, show_times=True, show_events=args.show_events)
        anim = animator.run()

        anim_path = f"animations/{file_index:03d}_{f_from}-{f_to}_fps{sampled_fps}.mp4"
        anim.save(anim_path, writer=writer)
