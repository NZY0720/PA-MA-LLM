from __future__ import annotations

from data.scenario_factory import export_weekly_low_carbon_scenario
from utils.constants import SCENARIO_PATH


def main() -> None:
    exported = export_weekly_low_carbon_scenario(SCENARIO_PATH)
    print(f"Scenario exported to: {exported}")


if __name__ == "__main__":
    main()
