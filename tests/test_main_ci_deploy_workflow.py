from pathlib import Path

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "main-ci-deploy.yml"


def read_workflow() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_push_trigger_ignores_docs_and_beads_only_commits():
    workflow = read_workflow()

    assert "paths-ignore:" in workflow
    assert "- 'docs/**'" in workflow
    assert "- '.beads/**'" in workflow
