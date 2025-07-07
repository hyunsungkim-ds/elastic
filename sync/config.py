LINEUP_PATH = "data/ajax/lineup/line_up.parquet"
EVENT_PATH = "data/ajax/event/event.parquet"
TRACKING_DIR = "data/ajax/tracking"
OUTPUT_DIR = "data/ajax/event_synced"

FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters

SPADL_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "goalkick",
    "shot_block",  # new, pass-like
    "ball_recovery",  # new, incoming
    "keeper_sweeper",  # new, incoming
    "dispossessed",  # new, minor
]
SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

PASS_LIKE_OPEN = ["pass", "cross", "shot", "clearance", "keeper_punch", "shot_block"]
SET_PIECE_OOP = ["throw_in", "goalkick", "corner_short", "corner_crossed"]
SET_PIECE = SET_PIECE_OOP + ["freekick_short", "freekick_crossed", "shot_freekick", "shot_penalty"]
INCOMING = ["interception", "keeper_save", "keeper_claim", "keeper_pick_up", "keeper_sweeper", "ball_recovery"]
MINOR = ["take_on", "second_take_on", "foul", "bad_touch", "dispossessed"]

TIME_PASS_LIKE_OPEN = 5  # unit: seconds
TIME_INCOMING = 5  # unit: seconds
TIME_SET_PIECE = 15  # unit: seconds, 10 in the ETSY paper
TIME_BAD_TOUCH = 5  # unit: seconds
TIME_FAULT_LIKE = 5  # unit: seconds

EVENT_COLS = [
    "frame",
    "period_id",
    "synced_ts",
    "utc_timestamp",
    "player_id",
    "object_id",
    "player_name",
    "advanced_position",
    "spadl_type",
    "outcome",
    "offside",
    "expected_goal",
]
NEXT_EVENT_COLS = ["next_player_id", "next_type", "receiver_id", "receive_frame"]
