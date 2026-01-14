import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np

from sync.elastic_nw import ELASTIC_NW
from tools.sportec_data import SportecData

if __name__ == "__main__":
    match_ids = np.sort([f.split(".")[0].split("-")[-1] for f in os.listdir("data/sportec/event")])
    OUTPUT_DIR = "data/sportec/event_synced"

    for i, match_id in enumerate(match_ids):
        print(f"\n[{i + 1}] {match_id}")

        match = SportecData(match_id)
        input_events = match.format_events_for_syncer()
        input_tracking = match.format_tracking_for_syncer()

        syncer = ELASTIC_NW(input_events, input_tracking)
        synced_events = syncer.run(syncer.events)

        not_synced = synced_events[synced_events["frame_id"].isna()]
        print(f"{len(not_synced)} events are not aligned.")

        synced_events.to_parquet(f"{OUTPUT_DIR}/{match_id}.parquet")
