from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
TERMS = (
    "Hypothesis", "Funnel", "Sharpe deflated", "Reserve", "Reserva 2023+ with cap 3", "C0", "M1",
    "N (attempts)", "Stages 0–3", "Candidate", "1/N", "Shadow (sombra local)", "Paper",
    "Fraccionable", "Cuenta cash", "Clean epoch (época limpia, `invalida_bug`)", "PDT", "Snapshot", "Equity snapshot", "Master accounts database", "Manifest",
    "Train, test, and reserve", "Buy and hold", "Intrabar", "Control (backtesting.py)", "Golden test", "Walk-forward", "Cuenta chica",
)


def test_glossary_entries_have_definition_example_and_source():
    glossary = (ROOT / "docs/manual/glossary.md").read_text()
    entries = re.findall(r"^### (.+)$", glossary, re.MULTILINE)
    assert all(f"### {term} —" in glossary for term in TERMS)
    assert len(entries) == len(TERMS)
    assert "## Bonds and markets" not in glossary
    for entry in entries:
        line = next(line for line in glossary.splitlines() if line.startswith(f"### {entry}"))
        assert "; for example," in line
        assert "](../" in line
