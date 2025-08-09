import numpy as np
from pandera import Check, Column, DataFrameSchema, Index

from sync import config

elastic_event_schema = DataFrameSchema(
    {
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "utc_timestamp": Column(np.dtype("datetime64[ns]")),
        "player_id": Column(object),
        "spadl_type": Column(str, Check(lambda s: s.isin(config.SPADL_TYPES))),
        "success": Column(bool),
        # "offside": Column(bool),
    },
    index=Index(int),
)

etsy_event_schema = DataFrameSchema(
    {
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "utc_timestamp": Column(np.dtype("datetime64[ns]")),
        "player_id": Column(object),
        "spadl_type": Column(str, Check(lambda s: s.isin(config.SPADL_TYPES))),
        "start_x": Column(float, Check(lambda s: (s >= 0) & (s <= config.PITCH_X))),
        "start_y": Column(float, Check(lambda s: (s >= 0) & (s <= config.PITCH_Y))),
        # "bodypart_id": Column(int, Check(lambda s: s.isin(range(len(config.SPADL_BODYPARTS))))),
    },
    index=Index(int),
)

synced_event_schema = DataFrameSchema(
    {
        "period_id": Column(int, Check(lambda s: s.isin([1, 2]))),
        "utc_timestamp": Column(np.dtype("datetime64[ns]")),
        "frame": Column(float, Check(lambda s: (s >= 0) & (round(s) == s)), nullable=True),
        "player_id": Column(object),
        "spadl_type": Column(str, Check(lambda s: s.isin(config.SPADL_TYPES))),
        "success": Column(bool),
        # "offside": Column(bool),
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
        "speed": Column(float),
        "accel_s": Column(float),  # Derivative of speed
        "accel_v": Column(float),  # Norm of the derivative of 2D velocity vector
    },
    index=Index(int),
)
