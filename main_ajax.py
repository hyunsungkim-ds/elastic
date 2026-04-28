import argparse
import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from pandera.errors import SchemaError

from sync import config, elastic_greedy, elastic_nw, unsynced
from tools.evaluate import collapse_events
from tools.stats_perform_data import StatsPerformData, find_spadl_event_types

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", type=str, default=None, choices=["elastic_nw", "elastic_greedy"])
    args = parser.parse_args()

    if args.sync == "elastic_nw":
        output_dir = "data/ajax/event_synced_nw"
    elif args.sync == "elastic_greedy":
        output_dir = "data/ajax/event_synced_greedy"
    else:
        output_dir = "data/ajax/event_unsynced"

    os.makedirs(output_dir, exist_ok=True)

    lineups = pd.read_parquet(config.LINEUP_PATH)
    events = pd.read_parquet(config.EVENT_PATH)
    events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"])
    events = events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True)
    events = find_spadl_event_types(events)

    match_ids = np.sort(events["stats_perform_match_id"].unique())
    erroneous_matches = []

    for i, match_id in enumerate(match_ids):
        if not os.path.exists(f"{config.TRACKING_DIR}/{match_id}.parquet"):
            continue

        match_tracking = pd.read_parquet(f"{config.TRACKING_DIR}/{match_id}.parquet")
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].set_index("player_id")
        match_events = events[
            (events["stats_perform_match_id"] == match_id)
            & (events["spadl_type"].notna())
            & (events["player_id"].notna())
        ].copy()

        try:
            match_date = match_events["game_date"].iloc[0]
            match_name = match_events["game"].iloc[0]
            print(f"\n[{i}] {match_id}: {match_name} on {match_date}")
        except IndexError:
            print(f"\n[{i}] {match_id}: No match date or name found in the event data.")
            continue

        match = StatsPerformData(match_lineup, match_events, match_tracking)
        input_events = match.format_events_for_syncer()
        input_tracking = match.format_tracking_for_syncer()
        output_path = f"{output_dir}/{match_id}.csv"

        try:
            if args.sync == "elastic_nw":
                syncer = elastic_nw.ELASTIC_NW(input_events, input_tracking)
            elif args.sync == "elastic_greedy":
                syncer = elastic_greedy.ELASTIC_Greedy(input_events, input_tracking)
            else:
                syncer = unsynced.Unsynced(input_events, input_tracking)

            synced_events = syncer.run()
            not_synced = synced_events[synced_events["frame_id"].isna()]
            print(f"{len(not_synced)}/{len(synced_events)} events not aligned.")

            if args.sync == "elastic_nw":
                synced_events = collapse_events(synced_events, input_tracking)

            synced_events.to_csv(output_path, index=False, encoding="utf-8")

        except (SchemaError, Exception) as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            erroneous_matches.append(f"[{i}] {match_id}: {match_name} on {match_date}")
            print(f"Skipped due to error: {e}")
            continue

    if erroneous_matches:
        print("\nWarning: The following matches were not saved due to errors:")
        for m in erroneous_matches:
            print(m)
