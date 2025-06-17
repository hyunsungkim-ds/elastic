"""Schemas for the event and tracking data."""

import numpy as np
from pandera import Check, Column, DataFrameSchema, Index

from sync import config

event_schema = DataFrameSchema(
    {
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "utc_timestamp": Column(np.dtype("datetime64[ns]")),
        "player_id": Column(object),
        "spadl_type": Column(str, Check(lambda s: s.isin(config.SPADL_TYPES))),
        "start_x": Column(float, Check(lambda s: (s >= 0) & (s <= config.FIELD_LENGTH))),
        "start_y": Column(float, Check(lambda s: (s >= 0) & (s <= config.FIELD_WIDTH))),
        # "bodypart_id": Column(int, Check(lambda s: s.isin(range(len(config.SPADL_BODYPARTS))))),
        "outcome": Column(bool),
        "offside": Column(bool),
    },
    index=Index(int),
)

synced_event_schema = DataFrameSchema(
    {
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "utc_timestamp": Column(np.dtype("datetime64[ns]")),
        "frame": Column(float, Check(lambda s: (s >= 0) & (round(s) == s)), nullable=True),
        # "seconds": Column(float, Check(lambda s: (s >= 0) & (round(s, 2) == s)), nullable=True),
        "player_id": Column(object),
        "spadl_type": Column(str, Check(lambda s: s.isin(config.SPADL_TYPES))),
        "outcome": Column(bool),
        "offside": Column(bool),
    },
    index=Index(int),
)

tracking_schema = DataFrameSchema(
    {
        "frame": Column(int, Check(lambda s: s >= 0)),
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "timestamp": Column(float),
        "utc_timestamp": Column(np.dtype("datetime64[ns]")),
        "player_id": Column(object, nullable=True),  # Mandatory for players (not ball)
        "ball": Column(bool),
        "x": Column(float),
        "y": Column(float),
        "z": Column(float, Check(lambda s: s >= 0), nullable=True),  # Mandatory for ball (not players)
        "accel": Column(float),
    },
    index=Index(int),
)
