from __future__ import annotations

import argparse
import json

from data.scenario_factory import export_weekly_low_carbon_scenario
from methods.case1 import run_case1
from methods.case2 import run_case2
from utils.constants import DEFAULT_MODEL, SCENARIO_PATH
from visualization.structure import render_default_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the modularized weekly case-study experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-scenario", help="Generate the weekly base scenario JSON.")
    build_parser.add_argument("--seed", type=int, default=2026)

    subparsers.add_parser("render", help="Render the structural and weekly profile figures.")

    case1_parser = subparsers.add_parser("case1", help="Run Case I.")
    case1_parser.add_argument("--repeats", type=int, default=3)
    case1_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    case1_parser.add_argument("--mock-llm", action="store_true")

    case2_parser = subparsers.add_parser("case2", help="Run Case II.")
    case2_parser.add_argument("--repeats", type=int, default=3)
    case2_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    case2_parser.add_argument("--mock-llm", action="store_true")

    args = parser.parse_args()
    if args.command == "build-scenario":
        path = export_weekly_low_carbon_scenario(SCENARIO_PATH, random_seed=args.seed)
        print(f"Scenario exported to: {path}")
    elif args.command == "render":
        render_default_visualizations()
        print("Visualization assets rendered.")
    elif args.command == "case1":
        result = run_case1(repeats=args.repeats, model=args.model, use_mock_llm=args.mock_llm)
        print(json.dumps(result["aggregated_results"], indent=2, ensure_ascii=False))
    elif args.command == "case2":
        result = run_case2(repeats=args.repeats, model=args.model, use_mock_llm=args.mock_llm)
        print(json.dumps(result["aggregated_results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
