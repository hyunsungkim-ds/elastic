<div align="center">
	<h1>
		ELASTIC
	</h1>
</div>

Source code for the paper **ELASTIC: Event-Tracking Data Synchronization in Soccer Without Annotated Event Locations** by Kim et al., 2025 (under review).

## Introduction
**ELASTIC (Event-Location-AgnoSTIC synchronizer)** is an algorithm for synchronizing event and tracking data in soccer. The source code is largely based on its previous work, [ETSY](https://github.com/ML-KULeuven/ETSY.git) (Van Roy et al., 2023), but the key difference is that our algorithm does not rely on event locations recorded in the event data, which are manually annotated and thus also prone to spatial errors.

Instead, ELASTIC leverages more subtle motion features such as ball acceleration and kick distance to precisely detect the moment of pass-like or incoming events, as well as the player-ball distance that ETSY used. Our experimental results demonstrate that it outperforms existing synchronizers by a large margin. You can refer to more details in the paper (which will be uploaded soon).

## Getting Started
You can install ELASTIC by cloning this repository. After installing the packages listed in `requirements.txt`, you can simply follow `tutorial.ipynb` with a properly placed pair of event and trackint data to synchronize it.

## Data Availability
Our code requires tracking data in the [kloppy](https://kloppy.pysport.org) format and event data in the [SPADL](https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html) format. However, the dataset used in this project is proprietary and cannot be publicly shared as it is an internal asset of the data provider.

If you have your own event and tracking datasets, you can use them for training and testing component models described below. In the current implementation, the data should be placed in the following paths:
- Tracking data: per-match Parquet files in `data/tracking` directory
- Event data: A single parquet file at `data/event/event.parquet`
- Match lineups: A single parquet file at `data/lineup/line_up.parquet`