from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SCENARIO_PATH = OUTPUTS_DIR / "base_weekly_scenario.json"
CASE1_OUTPUT_DIR = OUTPUTS_DIR / "case1"
CASE2_OUTPUT_DIR = OUTPUTS_DIR / "case2"
FIGURES_OUTPUT_DIR = OUTPUTS_DIR / "figures"
KEY_PATH = BASE_DIR / "key.txt"

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
HOURS_PER_WEEK = HOURS_PER_DAY * DAYS_PER_WEEK
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
