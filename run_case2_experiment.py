import argparse

from methods.case2 import run_case2
from utils.constants import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Case II experiments for multiple low-carbon parks.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated LLM negotiation runs.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument("--mock-llm", action="store_true", help="Use the built-in mock LLM negotiation generator.")
    parser.add_argument("--reuse-existing-traces", action="store_true", help="Reuse existing per-run LLM trace files instead of regenerating them.")
    args = parser.parse_args()
    results = run_case2(repeats=args.repeats, model=args.model, use_mock_llm=args.mock_llm, reuse_existing_traces=args.reuse_existing_traces)
    print("Baseline,System cost,System emission,Electricity trading,Carbon credit,Violation rate")
    for item in results["aggregated_results"]:
        metrics = item["metrics"]
        print(
            ",".join(
                [
                    item["baseline"],
                    str(metrics["total_system_operating_cost"]["mean"]),
                    str(metrics["total_system_carbon_emission"]["mean"]),
                    str(metrics["interpark_trading_volume"]["mean"]),
                    str(metrics["carbon_credit_trading_volume"]["mean"]),
                    str(metrics["constraint_violation_rate"]["mean"]),
                ]
            )
        )
    if "ablation_results" in results:
        print("")
        print("Ablation,System cost,System emission,Electricity trading,Carbon credit,Violation rate")
        for item in results["ablation_results"]:
            metrics = item["metrics"]
            print(
                ",".join(
                    [
                        item["baseline"],
                        str(metrics["total_system_operating_cost"]["mean"]),
                        str(metrics["total_system_carbon_emission"]["mean"]),
                        str(metrics["interpark_trading_volume"]["mean"]),
                        str(metrics["carbon_credit_trading_volume"]["mean"]),
                        str(metrics["constraint_violation_rate"]["mean"]),
                    ]
                )
            )


if __name__ == "__main__":
    main()
