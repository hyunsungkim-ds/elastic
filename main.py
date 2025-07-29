import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from pandera.errors import SchemaError

from sync import config, elastic
from sync.preprocessor import Preprocessor, find_spadl_event_types

if __name__ == "__main__":
    lineups = pd.read_parquet(config.LINEUP_PATH)
    events = pd.read_parquet(config.EVENT_PATH)
    events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"])
    events = events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True)

    # Find SPADL-style event types
    events = find_spadl_event_types(events)

    # Per-game event-tracking synchronization
    game_ids = np.sort(events["stats_perform_match_id"].unique())
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    erroneous_games = []

    for i, game_id in enumerate(game_ids):
        if not os.path.exists(f"{config.TRACKING_DIR}/{game_id}.parquet"):
            continue

        traces = pd.read_parquet(f"{config.TRACKING_DIR}/{game_id}.parquet")
        game_lineup = lineups.loc[lineups["stats_perform_match_id"] == game_id].set_index("player_id")
        game_events = events[
            (events["stats_perform_match_id"] == game_id)
            & (events["player_id"].notna())
            & (events["spadl_type"].notna())
        ].copy()

        try:
            game_date = game_events["game_date"].iloc[0]
            game_name = game_events["game"].iloc[0]
            print(f"\n[{i}] {game_id}: {game_name} on {game_date}")
        except IndexError:
            print(f"\n[{i}] {game_id}: No game date or name found in the event data.")
            continue

        # Formatting the event and tracking data for the syncer
        proc = Preprocessor(game_lineup, game_events, traces)
        input_events = proc.format_events_for_syncer()
        input_traces = proc.format_traces_for_syncer()

        # Applying ELASTIC to synchronize the event and tracking data
        output_path = f"{config.OUTPUT_DIR}/{game_id}.csv"
        try:
            syncer = elastic.ELASTIC(input_events, input_traces)
            syncer.run()
        except SchemaError:
            if os.path.exists(output_path):
                os.remove(output_path)
            erroneous_games.append(f"[{i}] {game_id}")
            print("Synchronization for this game was skipped due to potential errors.")
            continue

        proc.events[config.EVENT_COLS[:4]] = syncer.events[config.EVENT_COLS[:4]]
        proc.events[config.NEXT_EVENT_COLS] = syncer.events[config.NEXT_EVENT_COLS]
        output_events = proc.events[config.EVENT_COLS + config.NEXT_EVENT_COLS]

        synced_events = proc.events[proc.events["frame"].notna()]
        last_synced_event = synced_events.iloc[-1]
        last_synced_episode = syncer.frames.at[last_synced_event["frame"], "episode_id"]

        if last_synced_episode >= syncer.frames["episode_id"].max() - 1:
            output_events = output_events.loc[: last_synced_event.name]

        print(f"{len(synced_events)} events out of {len(output_events)} are synced.")

        if len(output_events) - len(synced_events) > 50:
            if os.path.exists(output_path):
                os.remove(output_path)
            erroneous_games.append(f"[{i}] {game_id}: {game_name} on {game_date}")
            print("The synced data file was not saved due to potential errors.")
        else:
            output_events.to_csv(output_path, index=False, encoding="utf-8")

    if erroneous_games:
        print("\nWarning: The following games were not saved due to potential errors:")
        for game_id in erroneous_games:
            print(game_id)
