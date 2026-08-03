from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return ROOT.joinpath(path).read_text(encoding="utf-8")


def test_readme_is_concise_and_decision_oriented():
    readme = _read("README.md")
    line_count = len(readme.splitlines())

    assert 150 <= line_count <= 220
    assert "pip install ntx" in readme
    assert "ntx solve --example --nu-hat 1e-2" in readme
    assert "From a source checkout" in readme
    assert "## Choose A Workflow" in readme
    assert "| Solved directly |" in readme
    assert "| Downstream closure |" in readme
    assert "| Validated comparisons |" in readme
    assert "| Research scope |" in readme
    # Three figures, and which three is the point: two validation panels plus
    # the measured gradient comparison the README leads with. Pinning the count
    # alone let a figure be swapped for an unrelated one.
    figures = re.findall(r"!\[[^]]+\]\(docs/_static/([^)]+)\.png\)", readme)
    assert sorted(figures) == [
        "bootstrap_current_fixed_field_validation",
        "design_derivatives",
        "validation_summary",
    ]
    assert re.search(r"Runtime\s+code does not use fitted\s+bridge constants", readme)


def test_docs_entry_point_displays_complete_normalized_dke():
    index = _read("docs/index.md")

    assert "\\xi \\frac{1}{B}" in index
    assert "+ \\frac{\\hat E_\\psi}" in index
    assert "- \\frac{1-\\xi^2}{2B^2}" in index
    assert "- C_L" in index
    assert "\\right] f = s" in index


def test_example_sections_have_unique_sequential_numbers():
    examples = _read("docs/examples.md")
    numbers = [
        int(match.group(1))
        for match in re.finditer(r"^## (\d+)\. ", examples, flags=re.MULTILINE)
    ]

    assert numbers == list(range(1, len(numbers) + 1))


def test_plan_owns_roadmap_and_release_checklist_is_release_only():
    roadmap = _read("docs/research-roadmap.md")
    checklist = _read("docs/ship-checklist.md")

    assert "private planning document" in roadmap
    assert "Current Audit Notes" not in checklist
    assert "Immediate Next Order" not in checklist
    assert "## Tag And Publish" in checklist


def test_api_and_glossary_are_in_the_documentation_tree():
    index = _read("docs/index.md")

    assert "- [API Reference](api.rst)" in index
    assert "- [Glossary](glossary.md)" in index
    assert "\napi\nglossary\n" in index
