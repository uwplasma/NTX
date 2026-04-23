from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_INPUT = ROOT / "coverage.json"
DEFAULT_JSON_OUTPUT = ROOT / "coverage-report.json"
DEFAULT_TEXT_OUTPUT = ROOT / "coverage-report.txt"


@dataclass(frozen=True)
class ModuleCoverage:
    module: str
    path: str
    percent_covered: float
    covered_lines: int
    missing_lines: int
    num_statements: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-input", type=Path, default=DEFAULT_JSON_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    return parser.parse_args()


def _module_name(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "src/ntx/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.removesuffix(".py").replace("/", ".")


def _is_ntx_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        normalized == "src/ntx.py"
        or normalized.startswith("src/ntx/")
        or "/src/ntx/" in normalized
    )


def _load_module_rows(payload: dict[str, object]) -> list[ModuleCoverage]:
    files = payload.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("coverage JSON is missing the 'files' mapping")

    rows: list[ModuleCoverage] = []
    for path, file_payload in files.items():
        if not isinstance(path, str) or not _is_ntx_source_path(path):
            continue
        if not isinstance(file_payload, dict):
            continue
        summary = file_payload.get("summary", {})
        if not isinstance(summary, dict):
            continue
        rows.append(
            ModuleCoverage(
                module=_module_name(path),
                path=path,
                percent_covered=float(summary.get("percent_covered", 0.0)),
                covered_lines=int(summary.get("covered_lines", 0)),
                missing_lines=int(summary.get("missing_lines", 0)),
                num_statements=int(summary.get("num_statements", 0)),
            )
        )

    rows.sort(key=lambda row: (row.percent_covered, row.module))
    return rows


def _render_text(overall: float, rows: list[ModuleCoverage]) -> str:
    lines = [
        "NTX Coverage Report",
        "===================",
        "",
        f"Overall coverage: {overall:.1f}%",
        "",
        "Module coverage:",
    ]
    for row in rows:
        lines.append(
            f"- {row.module}: {row.percent_covered:.1f}% "
            f"({row.covered_lines}/{row.num_statements} lines)"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(json_input: Path, json_output: Path, text_output: Path) -> dict[str, object]:
    payload = json.loads(json_input.read_text(encoding="utf-8"))
    totals = payload.get("totals", {})
    if not isinstance(totals, dict):
        raise ValueError("coverage JSON is missing the 'totals' summary")

    overall = float(totals.get("percent_covered", 0.0))
    rows = _load_module_rows(payload)
    report = {
        "overall_percent_covered": overall,
        "module_count": len(rows),
        "modules": [asdict(row) for row in rows],
    }

    json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text_output.write_text(_render_text(overall, rows), encoding="utf-8")
    return report


def main() -> int:
    args = _parse_args()
    build_report(
        json_input=args.json_input,
        json_output=args.json_output,
        text_output=args.text_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
