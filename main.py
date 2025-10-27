import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from pandera.errors import SchemaError

from sync import config, elastic
from tools.stats_perform_data import StatsPerformData, find_spadl_event_types

if __name__ == "__main__":
    LINEUP_PATH = "data/ajax/lineup/line_up_v3.parquet"
    EVENT_PATH = "data/ajax/event/event_v3.parquet"
    TRACKING_DIR = "data/ajax/tracking_v3"

    lineups = pd.read_parquet(LINEUP_PATH)
    events = pd.read_parquet(EVENT_PATH)
    events["utc_timestamp"] = pd.to_datetime(events["time_stamp"])
    events["game_date"] = events["utc_timestamp"].dt.date
    events = events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True)

    # Remove invalid matches with multiple dates
    match_dates = events[["stats_perform_match_id", "game_date"]].drop_duplicates()
    match_counts = match_dates["stats_perform_match_id"].value_counts()
    valid_matches = match_counts[match_counts == 1].index
    events = events[events["stats_perform_match_id"].isin(valid_matches)].copy()
    lineups = lineups[lineups["stats_perform_match_id"].isin(valid_matches)].copy()

    # Find necessary attributes for data preprocessing
    player_positions = lineups[["stats_perform_match_id", "player_id", "advanced_position"]].copy()
    events = pd.merge(events, player_positions)
    events = find_spadl_event_types(events)

    match_dates = events[["stats_perform_match_id", "game_date"]].drop_duplicates()
    match_dates = match_dates.set_index("stats_perform_match_id")["game_date"].to_dict()
    lineups["game_date"] = lineups["stats_perform_match_id"].map(match_dates)

    lineups_old = pd.read_parquet(config.LINEUP_PATH)
    team_names = lineups_old[["contestant_id", "contestant_name"]].drop_duplicates()
    team_names = team_names.set_index("contestant_id")["contestant_name"].to_dict()
    lineups["contestant_name"] = lineups["contestant_id"].map(team_names)

    # Find SPADL-style event types
    events = find_spadl_event_types(events)

    # Per-match event-tracking synchronization
    match_ids = np.sort(events["stats_perform_match_id"].unique())
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    erroneous_matches = []

    for i, match_id in enumerate(match_ids):
        tracking_path = f"{TRACKING_DIR}/{match_id}_new.parquet"
        output_path = f"{config.OUTPUT_DIR}/{match_id}.csv"

        if not os.path.exists(tracking_path) and os.path.exists(output_path):
            continue

        match_tracking = pd.read_parquet(f"{TRACKING_DIR}/{match_id}_new.parquet")
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].set_index("player_id")
        match_events = events[
            (events["stats_perform_match_id"] == match_id)
            & (events["player_id"].notna())
            & (events["spadl_type"].notna())
        ].copy()

        try:
            match_date = match_events["game_date"].iloc[0]
            match_name = " vs ".join(match_lineup["contestant_name"].unique())
            print(f"\n[{i}] {match_id}: {match_name} on {match_date}")
        except IndexError:
            print(f"\n[{i}] {match_id}: No match date or name found in the event data.")
            continue

        # Formatting the event and tracking data for the syncer
        match = StatsPerformData(match_lineup, match_events, match_tracking)
        input_events = match.format_events_for_syncer()
        input_tracking = match.format_tracking_for_syncer()

        # Applying ELASTIC to synchronize the event and tracking data
        try:
            syncer = elastic.ELASTIC(input_events, input_tracking)
            syncer.run()
        except SchemaError:
            if os.path.exists(output_path):
                os.remove(output_path)
            erroneous_matches.append(f"[{i}] {match_id}: {match_name} on {match_date}")
            print("Synchronization for this match was skipped due to potential errors.")
            continue

        match.events[config.EVENT_COLS[:4]] = syncer.events[config.EVENT_COLS[:4]]
        match.events[config.NEXT_EVENT_COLS] = syncer.events[config.NEXT_EVENT_COLS]
        output_events = match.events[config.EVENT_COLS + config.NEXT_EVENT_COLS]

        synced_events = match.events[match.events["frame_id"].notna()]
        last_synced_event = synced_events.iloc[-1]
        last_synced_episode = syncer.frames.at[last_synced_event["frame_id"], "episode_id"]

        if last_synced_episode >= syncer.frames["episode_id"].max() - 1:
            output_events = output_events.loc[: last_synced_event.name]

        print(f"{len(synced_events)} events out of {len(output_events)} are synced.")

        if len(output_events) - len(synced_events) > 50:
            if os.path.exists(output_path):
                os.remove(output_path)
            erroneous_matches.append(f"[{i}] {match_id}: {match_name} on {match_date}")
            print("The synced data file was not saved due to potential errors.")
        else:
            output_events.to_csv(output_path, index=False, encoding="utf-8")

    if erroneous_matches:
        print("\nWarning: The following matches were not saved due to potential errors:")
        for match_id in erroneous_matches:
            print(match_id)
