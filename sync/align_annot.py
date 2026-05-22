"""Align per-annotator event CSVs into a single cross-rater table per match."""

import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from itertools import combinations

import numpy as np
import pandas as pd

ANNOTATORS = ["hyunsung", "hoyoung", "kunhee"]
MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY"]
DATA_DIR = "data/sportec/event_corrected"

DROP_COLS = ["synced_frame_id", "receiver_id", "receive_ts", "receive_frame_id", "id"]
KEY_COLS = ["period_id", "player_id", "spadl_type", "outcome"]
SPLIT_ERROR_TYPES = {"player_id", "spadl_type", "false_positive", "missing", "outcome"}

TS_DIFF_THRESHOLD_SEC = 2
TS_DIFF_SYNCED_SEC = 10
LOOKAHEAD = 30

OUTPUT_COLS = KEY_COLS + [f"{a}_ts" for a in ANNOTATORS] + ["error_type", "note"]


def load_annot(match_id, annotator):
    df = pd.read_csv(f"{DATA_DIR}/{match_id}_{annotator}.csv")
    return df.drop(columns=DROP_COLS).reset_index(drop=True)


def combine_notes(notes_by_ann):
    parts = [f"{ann}: {note}" for ann, note in notes_by_ann.items() if pd.notna(note) and str(note).strip() != ""]
    return "\n".join(parts) if parts else np.nan


def ts_to_sec(ts):
    if pd.isna(ts):
        return np.inf
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def row_key(row):
    return tuple(row[c] for c in KEY_COLS)


def match(row_a, row_b):
    if row_key(row_a) != row_key(row_b):
        return False
    ts_a = ts_to_sec(row_a["synced_ts"])
    ts_b = ts_to_sec(row_b["synced_ts"])
    if np.isinf(ts_a) or np.isinf(ts_b):
        return True
    threshold = TS_DIFF_THRESHOLD_SEC
    if row_a["error_type"] == "synced_ts" or row_b["error_type"] == "synced_ts":
        threshold = TS_DIFF_SYNCED_SEC
    return abs(ts_a - ts_b) <= threshold


def make_merge_row(rows_by_ann):
    first = next(iter(rows_by_ann.values()))
    has_synced_err = any(r["error_type"] == "synced_ts" for r in rows_by_ann.values())
    rec = {
        "period_id": first["period_id"],
        "player_id": first["player_id"],
        "spadl_type": first["spadl_type"],
        "outcome": first["outcome"],
        "error_type": "synced_ts" if has_synced_err else np.nan,
        "note": combine_notes({a: r["note"] for a, r in rows_by_ann.items()}),
    }
    for ann in ANNOTATORS:
        rec[f"{ann}_ts"] = rows_by_ann[ann]["synced_ts"] if ann in rows_by_ann else np.nan
    return rec


def make_split_row(ann, row, implicit=False):
    err = "attr_mismatch" if implicit else row["error_type"]
    rec = {
        "period_id": row["period_id"],
        "player_id": row["player_id"],
        "spadl_type": row["spadl_type"],
        "outcome": row["outcome"],
        "error_type": err,
        "note": combine_notes({ann: row["note"]}),
    }
    for other in ANNOTATORS:
        rec[f"{other}_ts"] = row["synced_ts"] if other == ann else np.nan
    return rec


def align_annots(match_id):
    dfs = {a: load_annot(match_id, a) for a in ANNOTATORS}
    ptr = {a: 0 for a in ANNOTATORS}

    def in_range(a):
        return ptr[a] < len(dfs[a])

    def row(a):
        return dfs[a].iloc[ptr[a]] if in_range(a) else None

    def earliest(active):
        return min(active, key=lambda a: (row(a)["period_id"], ts_to_sec(row(a)["synced_ts"])))

    def find_forward(ann, target_row):
        target_period = target_row["period_id"]
        for la in range(ptr[ann] + 1, min(ptr[ann] + 1 + LOOKAHEAD, len(dfs[ann]))):
            la_row = dfs[ann].iloc[la]
            if la_row["period_id"] != target_period:
                return None
            if match(la_row, target_row):
                return la
        return None

    records = []

    while any(in_range(a) for a in ANNOTATORS):
        active = [a for a in ANNOTATORS if in_range(a)]

        # Step 1: explicit split error 우선 소진
        progressed = False
        for a in active:
            if row(a)["error_type"] in SPLIT_ERROR_TYPES:
                records.append(make_split_row(a, row(a), implicit=False))
                ptr[a] += 1
                progressed = True
                break
        if progressed:
            continue

        # Step 1.5: ts=NaN row 단독 split
        for a in active:
            if pd.isna(row(a)["synced_ts"]):
                rec = make_split_row(a, row(a), implicit=False)
                rec["error_type"] = "ts_missing"
                records.append(rec)
                ptr[a] += 1
                progressed = True
                break
        if progressed:
            continue

        if len(active) == 1:
            a = active[0]
            records.append(make_split_row(a, row(a), implicit=True))
            ptr[a] += 1
            continue

        if len(active) == 2:
            a, b = active
            if match(row(a), row(b)):
                records.append(make_merge_row({a: row(a), b: row(b)}))
                ptr[a] += 1
                ptr[b] += 1
            else:
                t = earliest(active)
                other = b if t == a else a
                fwd_other = find_forward(other, row(t))
                if fwd_other is not None:
                    for skip in range(ptr[other], fwd_other):
                        records.append(make_split_row(other, dfs[other].iloc[skip], implicit=True))
                    ptr[other] = fwd_other
                else:
                    records.append(make_split_row(t, row(t), implicit=True))
                    ptr[t] += 1
            continue

        # active == 3
        pairs = [(a, b) for a, b in combinations(active, 2) if match(row(a), row(b))]

        if len(pairs) == 3:
            records.append(make_merge_row({a: row(a) for a in active}))
            for a in active:
                ptr[a] += 1
            continue

        if len(pairs) >= 1:
            a1, a2 = pairs[0]
            odd = [a for a in active if a not in (a1, a2)][0]

            fwd_maj = find_forward(a1, row(odd))
            if fwd_maj is not None:
                records.append(make_merge_row({a1: row(a1), a2: row(a2)}))
                ptr[a1] += 1
                ptr[a2] += 1
                continue

            fwd_odd = find_forward(odd, row(a1))
            if fwd_odd is not None:
                for skip in range(ptr[odd], fwd_odd):
                    records.append(make_split_row(odd, dfs[odd].iloc[skip], implicit=True))
                ptr[odd] = fwd_odd
                continue

            records.append(make_split_row(odd, row(odd), implicit=True))
            ptr[odd] += 1
            continue

        t = earliest(active)
        records.append(make_split_row(t, row(t), implicit=True))
        ptr[t] += 1

    df = pd.DataFrame(records)
    ts_cols = [f"{a}_ts" for a in ANNOTATORS]
    secs = df[ts_cols].map(ts_to_sec).replace(np.inf, np.nan)
    df = df.assign(_med=secs.median(axis=1).fillna(np.inf))
    df = df.sort_values(["period_id", "_med"], kind="stable").drop(columns="_med").reset_index(drop=True)
    return df[OUTPUT_COLS]


if __name__ == "__main__":
    for match_id in MATCH_IDS:
        aligned = align_annots(match_id)
        out_path = f"{DATA_DIR}/{match_id}_aligned.csv"
        aligned.to_csv(out_path, index=False)
        print(f"Saved {out_path}: {len(aligned)} rows")
