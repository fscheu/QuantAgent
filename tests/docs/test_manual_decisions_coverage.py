from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_decision_index_links_every_decision_document():
    index = (ROOT / "docs/manual/decisions.md").read_text()
    for decision in (ROOT / "docs/04_decisions").glob("*.md"):
        assert f"../04_decisions/{decision.name}" in index
