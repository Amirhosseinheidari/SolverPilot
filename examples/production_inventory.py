"""Run the two-period inventory/CVaR/PWL example and its independent oracle."""
import json
from solverpilot.applications.inventory import solve_inventory_example

if __name__ == '__main__':
    report = solve_inventory_example()
    print(json.dumps(report, indent=2))
    if not report.get('reference_agrees'):
        raise SystemExit(1)
