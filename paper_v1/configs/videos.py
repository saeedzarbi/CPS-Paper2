"""
Canonical evaluation video set (v19): 12 publicly available reference sequences.

Split
-----
- TRAIN_VIDEOS (9): used for PPO training and CPS train/eval breakdown
- HELD_OUT_VIDEOS (3): never seen during PPO training; generalization test

Sources: six Blender open movies + six Xiph DERF clips (1080p or 720p).

Notes
-----
- ``tearsofsteel`` replaces legacy ``tearsofsteel_short`` (120 s clip).
- ``sunflower`` stands in for the originally proposed ``umbrella`` (no DERF clip).
- ``kristen_and_sara`` stands in for ``meridian`` (Netflix Meridian IMF is ~96 GB;
  Kristen & Sara is the standard DERF talking-head held-out substitute).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, TypedDict

# Project root (new/) — enables ``from configs.videos import ...`` when run as script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paths import get_paths  # noqa: E402

_P = get_paths()
RAW_VIDEOS_DIR: Path = _P["raw_videos"]
ENCODED_VIDEOS_DIR: Path = _P["encoded_videos"]
VMAF_SCORES_DIR: Path = _P["vmaf_scores"]
CONTENT_FEATURES_DIR: Path = _P["content_features"]


class VideoSpec(TypedDict):
    slug: str
    display_name: str
    source: str          # "blender" | "derf"
    ladder_role: str     # saturation | steep | flat | mixed | talking_head
    held_out: bool


# ---------------------------------------------------------------------------
# Final 12-video roster (approved)
# ---------------------------------------------------------------------------

VIDEO_SPECS: List[VideoSpec] = [
    # --- Blender (6) ---
    {"slug": "bigbuckbunny", "display_name": "Big Buck Bunny", "source": "blender",
     "ladder_role": "saturation", "held_out": False},
    {"slug": "tearsofsteel", "display_name": "Tears of Steel", "source": "blender",
     "ladder_role": "saturation", "held_out": False},
    {"slug": "sintel", "display_name": "Sintel", "source": "blender",
     "ladder_role": "mixed", "held_out": True},
    {"slug": "elephants_dream", "display_name": "Elephants Dream", "source": "blender",
     "ladder_role": "saturation", "held_out": False},
    {"slug": "ducks", "display_name": "Ducks Take Off", "source": "derf",
     "ladder_role": "steep", "held_out": False},
    {"slug": "cosmos", "display_name": "Cosmos Laundromat", "source": "blender",
     "ladder_role": "mixed", "held_out": True},
    # --- Xiph DERF (6) ---
    {"slug": "parkjoy", "display_name": "Park Joy", "source": "derf",
     "ladder_role": "steep", "held_out": False},
    {"slug": "old_town_cross", "display_name": "Old Town Cross", "source": "derf",
     "ladder_role": "flat", "held_out": False},
    {"slug": "rush_hour", "display_name": "Rush Hour", "source": "derf",
     "ladder_role": "steep", "held_out": False},
    {"slug": "into_tree", "display_name": "Into Tree", "source": "derf",
     "ladder_role": "flat", "held_out": False},
    {"slug": "sunflower", "display_name": "Sunflower", "source": "derf",
     "ladder_role": "flat", "held_out": False},
    {"slug": "kristen_and_sara", "display_name": "Kristen & Sara", "source": "derf",
     "ladder_role": "talking_head", "held_out": True},
]

ALL_VIDEOS: List[str] = [v["slug"] for v in VIDEO_SPECS]

TRAIN_VIDEOS: List[str] = [v["slug"] for v in VIDEO_SPECS if not v["held_out"]]

HELD_OUT_VIDEOS: List[str] = [v["slug"] for v in VIDEO_SPECS if v["held_out"]]

# CPS / shield eval uses all 12 titles
EVAL_VIDEOS: List[str] = list(ALL_VIDEOS)

# Legacy aliases (pre-v19 scripts / CSV rows)
LEGACY_ALIASES: Dict[str, str] = {
    "tearsofsteel_short": "tearsofsteel",
}

REMOVED_VIDEOS: frozenset[str] = frozenset({"crowd_run"})

# Paper-facing display names
DISPLAY_NAMES: Dict[str, str] = {v["slug"]: v["display_name"] for v in VIDEO_SPECS}

# Eval protocol
SEEDS_PER_VIDEO: int = 17          # 12 * 17 = 204 ≈ 200 episodes
CPS_EPISODES: int = SEEDS_PER_VIDEO * len(ALL_VIDEOS)
MAX_CHUNKS: int = 48               # 192 s @ 4 s/chunk

# Comma-separated default for CLI flags
EVAL_VIDEOS_CSV: str = ",".join(EVAL_VIDEOS)
TRAIN_VIDEOS_CSV: str = ",".join(TRAIN_VIDEOS)
HELD_OUT_VIDEOS_CSV: str = ",".join(HELD_OUT_VIDEOS)


def resolve_slug(name: str) -> str:
    """Map legacy slugs to the v19 canonical name."""
    return LEGACY_ALIASES.get(name, name)


if __name__ == "__main__":
    print(f"v19 video set: {len(ALL_VIDEOS)} titles")
    print(f"  train/eval ({len(TRAIN_VIDEOS)}): {TRAIN_VIDEOS_CSV}")
    print(f"  held-out  ({len(HELD_OUT_VIDEOS)}): {HELD_OUT_VIDEOS_CSV}")
    print(f"  CPS episodes (default): {CPS_EPISODES}")
