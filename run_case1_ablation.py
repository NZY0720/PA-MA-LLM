from methods.case1 import run_case1_ablation_from_existing


def main() -> None:
    results = run_case1_ablation_from_existing()
    print("Variant,Total cost,Carbon emission,High-carbon purchase,Fairness index,Violation rate,Cost std")
    for item in results:
        metrics = item["metrics"]
        print(
            ",".join(
                [
                    item["baseline"],
                    str(metrics["total_operating_cost"]["mean"]),
                    str(metrics["total_carbon_emission"]["mean"]),
                    str(metrics["high_carbon_grid_purchase"]["mean"]),
                    str(metrics["carbon_fairness_index"]["mean"]),
                    str(metrics["constraint_violation_rate"]["mean"]),
                    str(metrics["total_operating_cost"]["std"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
