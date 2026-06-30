import os

# --- game ---
GAME        = "Breakout"
FRAME_STACK = 4

# --- training ---
MAX_STEPS        = 50_000_000
REPLAY_SIZE      = 500_000
BATCH_SIZE       = 32
LEARNING_RATE    = 0.00025
GAMMA            = 0.99
EPSILON_START    = 1.0
EPSILON_END      = 0.1
EPSILON_DECAY    = 1_000_000
MIN_REPLAY_SIZE  = 50_000
TARGET_SYNC_FREQ = 10_000
CHECKPOINT_FREQ  = 100_000

# --- paths ---
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VIDEO_DIR      = os.path.join(BASE_DIR, "videos")
