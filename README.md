<h1 align="center">ELASTIC</h1>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2608.30227-b31b1b)](https://arxiv.org/abs/2608.30227)
[![Code: MPL 2.0](https://img.shields.io/badge/Code-MPL%202.0-brightgreen)](https://opensource.org/licenses/MPL-2.0)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-brightgreen)](https://creativecommons.org/licenses/by/4.0/)

</div>

Source code for the paper [ELASTIC: Trajectory-Based Synchronization of Event and Tracking Data in Soccer](https://arxiv.org/abs/2608.30227) by Kim et al., CIKM 2026.

## Introduction
**ELASTIC (Event-Location-AgnoSTIC synchronizer)** is an algorithm for synchronizing event and tracking data in soccer. Unlike prior synchronizers such as [ETSY (Van Roy et al., 2023)](https://link.springer.com/chapter/10.1007/978-3-031-53833-9_2) and [DataBallPy (Oonk et al., 2025)](https://www.researchgate.net/publication/394432453_The_Right_Way_to_Synchronize_Tracking_and_Event_Data_Using_Domain_Knowledge_to_Optimize_Algorithms), it does not rely on human-annotated event locations, which are themselves prone to spatial errors. Instead, it infers the start and end timestamps of each event solely from player and ball trajectories.

To this end, ELASTIC first extracts a sparse set of *candidate frames* where a ball touch is physically plausible, using motion features such as ball acceleration, player-ball distance, and kick distance. It then aligns the event sequence with the candidate-frame sequence using the [Needleman-Wunsch algorithm (Needleman & Wunsch, 1970)](https://doi.org/10.1016/0022-2836(70)90057-4), preserving the order of events.

As a visual result, the following video compares the raw event timestamps/locations (black "x") and the synchronized event timestamps/locations (orange "★"), alongside player and ball trajectories.
<p align="center">
  <img src="docs/sportec_J03WMX_1_0015-0035.gif"/>
</p>

## Instructions and Visualizations
First, clone this repository and install the packages listed in `requirements.txt`.

Next, download the [Sportec Open DFL Dataset (Bassek et al., 2025)](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177) and place its XML files under `data/sportec/metadata`, `data/sportec/event`, and `data/sportec/tracking`. See the *Data Preparation* part of `tutorial.ipynb` for details.

Then, follow `tutorial.ipynb`, which applies ELASTIC to that dataset. In particular, the last part of the notebook lets you visualize the three steps of ELASTIC for a given window as follows:

### 1. Player-ball distances with candidate frames (Section 2.2 & Fig. 2)
ELASTIC first narrows the search space by keeping only the frames where a ball touch is physically plausible. In the figure below, each colored curve is the distance between the ball and one player in the window. The black dashed lines are the extracted candidate frames, and the red solid lines are the synchronized events, labeled `P` for a pass and `C` for a ball control.
<p align="center">
  <img src="docs/cand_frames.png"/>
</p>

### 2. Pairwise scores (Section 2.3 & Fig. 3a)
Within each in-play segment, ELASTIC scores every (event, candidate frame) pair using features such as ball acceleration, player-ball distance, and kick distance. In the figure below, each cell is the pairwise score between an event (row) and a candidate frame (column), with a darker color indicating a better match.
<p align="center">
  <img src="docs/score_mat.png"/>
</p>

### 3. DP table for the NW alignment (Section 2.4 & Fig. 3b)
Finally, the Needleman-Wunsch algorithm picks the highest-scoring assignment that keeps the events in chronological order. In the figure below, each cell of the dynamic programming (DP) table holds the cumulative score and the move that produced it. The optimal path traced back from the bottom-right corner is highlighted in yellow, and the frame IDs it selects are marked in red.
<p align="center">
  <img src="docs/dp_table.png"/>
</p>


## Reproducing the Experiments
The scripts under `experiments/` reproduce the experimental results on the three re-annotated Sportec matches (`J03WMX`, `J03WN1`, `J03WPY`). Run them from the repository root.

### 1. Benchmark data construction (Section 3.1 & Table 1)
Reproducing the experiments needs two inputs:

- **Re-annotated event data** already contained in `benchmark/` of this repository.
- **Tracking data** that can be downloaded from [this link](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177), following the *Data Preparation* part of `tutorial.ipynb`. In particular, the tracking XML files should be placed under `data/sportec/tracking`.

During the re-annotation process, we removed the false-positive events and inserted the missing ones, so that the evaluation can focus solely on the synchronization performance. The corrected events, with only their timestamps left unsynchronized, are placed in `benchmark/unsynced/` and used as the input to the synchronizers.

`benchmark/gt/` then holds the same events under the same indexing, with our annotated timestamps attached: `frame_id` for the moment the event occurs and `receive_frame_id` for the moment the ball is received.

`benchmark.py` is the script that built these two files: it reads the per-annotator labels, measures the inter-annotator reliability reported in Table 1, and takes the median of the three annotations as the ground truth. We do not publish the per-annotator labels, so this script is included for reference only.

```bash
python experiments/benchmark.py
```

### 2. Synchronization accuracy (Sections 3.2–3.3 & Table 2)
`evaluate.py` synchronizes the unsynced events, compares the result against the ground truth, and reports the accuracy metrics in Table 2. It takes the following arguments:
- `--method` selects the synchronizer among `etsy`, `biermann`, `databallpy`, `elastic_greedy`, and `elastic_nw`.
- `--save` caches the synchronized events to `benchmark/synced/{method}/{match_id}.parquet`.
- `--load` reads that cache back instead of running the synchronizer again. `databallpy` always does so, since its outputs come from the external DataBallPy package.

```bash
python experiments/evaluate.py --method elastic_nw [--save | --load]
```

### 3. Candidate frame selection (Section 3.4 & Table 3)
`cand_sweep.py` varies the candidate frame detection conditions and thresholds, and reports the resulting candidate frame coverage and the synchronization accuracy in Table 3.

```bash
python experiments/cand_sweep.py
```

### 4. Hyperparameter sensitivity (Section 3.5 & Tables 4–5)
`coeff_sweep.py` varies the weights of per-feature scores (Table 4), and `score_sweep.py` varies the clipping bounds of the scoring functions and the alignment penalties (Table 5).

```bash
python experiments/coeff_sweep.py
python experiments/score_sweep.py
```

Every sweep writes one row per (config, match, event category) to `experiments/results/{script}_{timestamp}.csv`.

## Citation
If you use this code or the benchmark event data in your research, please consider citing our paper:
```bibtex
@inproceedings{kim2026elastic,
  author       = {Hyunsung Kim and
                  Hoyoung Choi and
                  Kunhee Lee and
                  Sangwoo Seo and
                  Tom Boomstra and
                  Jinsung Yoon and
                  Chanyoung Park},
  title        = {{ELASTIC}: Trajectory-Based Synchronization of Event and Tracking Data in Soccer},
  booktitle    = {Proceedings of the 35th {ACM} International Conference on Information and Knowledge Management},
  year         = {2026},
  doi          = {10.1145/3799682.3841035},
}
```

## License and Attribution
The source code is released under the [MPL-2.0 license](LICENSE).

The event data under `benchmark/` is derived from the [Sportec Open DFL Dataset (Bassek et al., 2025)](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177), which is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with the authorization of the Deutsche Fussball Liga (DFL). We redistribute it under the same license, with the modifications described above (corrected event records and re-annotated timestamps). Please cite the original dataset alongside our paper:

```bibtex
@article{bassek2025integrated,
  author       = {Manuel Bassek and
                  Robert Rein and
                  Hendrik Weber and
                  Daniel Memmert},
  title        = {An integrated dataset of spatiotemporal and event data in elite soccer},
  journal      = {Scientific Data},
  volume       = {12},
  number       = {195},
  year         = {2025},
  doi          = {10.1038/s41597-025-04505-y},
}
```
