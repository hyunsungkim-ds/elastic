"""Defines ETSY configuration."""

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
    "ball_recovery",  # new, incoming-like
    "ball_touch",  # new, not handled
    "dispossessed",  # new, not handled
    "shot_block",  # new, pass-like
    "keeper_sweeper",  # new, incoming-like
]
SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

PASS_LIKE_OPEN = ["pass", "cross", "shot", "shot_block", "clearance", "keeper_punch"]
INCOMING = ["interception", "ball_recovery", "keeper_save", "keeper_claim", "keeper_pick_up", "keeper_sweeper"]
SET_PIECE_OOP = ["throw_in", "corner_crossed", "corner_short", "goalkick"]
SET_PIECE = SET_PIECE_OOP + ["freekick_crossed", "freekick_short", "shot_freekick", "shot_penalty"]

BAD_TOUCH = ["bad_touch"]
FAULT_LIKE = ["foul", "tackle"]
DUEL_LIKE = ["take_on", "dispossessed", "tackle"]


TIME_PASS_LIKE_OPEN = 5  # unit: seconds
TIME_INCOMING = 5  # unit: seconds
TIME_SET_PIECE = 15  # unit: seconds, 10 in the ETSY paper
TIME_BAD_TOUCH = 5  # unit: seconds
TIME_FAULT_LIKE = 5  # unit: seconds

EVENT_COLS = [
    "frame",
    "period_id",
    "seconds",
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
