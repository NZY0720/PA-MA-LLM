from __future__ import annotations

import argparse
import json

from methods.case1 import run_case1
from utils.constants import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Case I experiments for the single low-carbon park.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated LLM runs.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument("--mock-llm", action="store_true", help="Use the built-in mock LLM intent generator instead of calling DeepSeek.")
    args = parser.parse_args()
    results = run_case1(repeats=args.repeats, model=args.model, use_mock_llm=args.mock_llm)
    print(json.dumps(results["aggregated_results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
