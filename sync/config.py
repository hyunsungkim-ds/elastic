MATCHES_PATH = "data/ajax/matches.parquet"
LINEUP_PATH = "data/ajax/lineup/line_up.parquet"
EVENT_PATH = "data/ajax/event/event.parquet"
TRACKING_DIR = "data/ajax/tracking"
OUTPUT_DIR = "data/ajax/event_synced"

PITCH_X = 105.0  # unit: meters
PITCH_Y = 68.0  # unit: meters

SPADL_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "second_take_on",
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
    "control",  # new, incoming
    "out",  # new, virtual OOP marker
]
SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

# Event categories and parameters for ELASTIC
PASS_LIKE_OPEN = ["pass", "cross", "clearance", "shot", "shot_block", "keeper_punch", "bad_touch"]
SET_PIECE_OOP = ["throw_in", "goalkick", "corner_short", "corner_crossed"]
SET_PIECE = SET_PIECE_OOP + ["freekick_short", "freekick_crossed", "shot_freekick", "shot_penalty"]
INCOMING_GK = ["keeper_save", "keeper_claim", "keeper_pick_up", "keeper_sweeper"]
INCOMING = INCOMING_GK + ["interception", "ball_recovery"]
MINOR = ["tackle", "dispossessed", "take_on", "second_take_on", "foul"]
EVENT_END = ["control", "out", "goal"]

EVENT_CAT_MAP = (
    {x: "pass_like" for x in PASS_LIKE_OPEN}
    | {x: "set_piece" for x in SET_PIECE}
    | {x: "incoming" for x in INCOMING}
    | {x: "minor" for x in ["tackle", "dispossessed"]}
    | {x: "event_end" for x in EVENT_END if x != "foul"}
)

TIME_KICKOFF = 5
TIME_PASS_LIKE_OPEN = 5
TIME_SET_PIECE = 15
TIME_INCOMING = 5
TIME_MINOR = 5

# Additional event categories and parameters for ETSY
BAD_TOUCH = ["bad_touch"]
FAULT_LIKE = ["foul", "tackle", "dispossessed"]

TIME_BAD_TOUCH = 5
TIME_FAULT_LIKE = 5

ALIGNED_COLS = ["frame_id", "period_id", "episode_id", "timestamp", "player_id", "spadl_type", "success", "score"]
