"""Compute behavior fidelity for the C5 parameterized ablation.

The main pipeline computes fidelity only for B4'/B5/B6. C5 (parameterized
bidding) is needed as a contrast in Table VIII. Since C5 is deterministic,
fidelity computed on run_1 represents all runs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from methods.behavior_metrics import compute_behavior_fidelity
from methods.case2 import _build_park_states, _attach_subjective_profiles
from utils.io_utils import load_json

states = _build_park_states()
_attach_subjective_profiles(states, client=None, use_mock=True)
intent_path = ROOT / "outputs" / "case2" / "llm_traces" / "parameterized_ablation_run_1.json"
intent = load_json(intent_path)
fidelity = compute_behavior_fidelity(states, intent)
print(json.dumps(fidelity.as_dict(), indent=2))
