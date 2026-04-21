from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the current NTX physics gates from tracked artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="NTX repository root",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional path for a machine-readable report",
    )
    return parser.parse_args()


def main() -> None:
    from ntx.physics_gates import evaluate_artifact_gates, physics_gate_registry

    args = parse_args()
    root = args.root.resolve()
    artifact_results = {result.gate.name: result for result in evaluate_artifact_gates(root)}

    rows: list[dict[str, object]] = []
    for gate in physics_gate_registry():
        result = artifact_results.get(gate.name)
        status = result.status if result is not None else "test-backed"
        value = result.value if result is not None else None
        details = result.details if result is not None else gate.source
        rows.append(
            {
                "name": gate.name,
                "category": gate.category,
                "metric": gate.metric,
                "relation": gate.relation,
                "threshold": gate.threshold,
                "status": status,
                "value": value,
                "source": gate.source,
                "rationale": gate.rationale,
                "details": details,
            }
        )

    print("NTX physics gates")
    for row in rows:
        threshold = (
            f" {row['relation']} {row['threshold']:.3e}"
            if isinstance(row["threshold"], float)
            else ""
        )
        value = (
            f", value={row['value']:.3e}"
            if isinstance(row["value"], float)
            else ""
        )
        print(
            f"- {row['name']} [{row['category']}] {row['status']}: "
            f"{row['metric']}{threshold}{value}"
        )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps({"root": str(root), "gates": rows}, indent=2))


if __name__ == "__main__":
    main()
