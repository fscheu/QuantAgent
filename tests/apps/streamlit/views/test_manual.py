from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from apps.streamlit.views import manual


def test_rewrite_markdown_links_rewrites_relative_docs_links() -> None:
    current_doc = manual.DOCS_ROOT / "user-manual/index.md"
    markdown = (
        "[Guide](getting-started.md) "
        "[Paper](monitoring.md#paper-trading-tab) "
        "[External](https://example.com)"
    )

    rewritten = manual._rewrite_markdown_links(markdown, current_doc)

    assert "?view=User%20Manual&manual=user-manual/getting-started.md" in rewritten
    assert (
        "?view=User%20Manual&manual=user-manual/monitoring.md&manual_anchor=paper-trading-tab"
        in rewritten
    )
    assert "[External](https://example.com)" in rewritten


def test_rewrite_markdown_links_marks_missing_docs_as_unavailable() -> None:
    current_doc = manual.DOCS_ROOT / "user-manual/index.md"

    rewritten = manual._rewrite_markdown_links("[Missing](missing.md)", current_doc)

    assert rewritten == "Missing (`missing.md` unavailable in app)"


def test_resolve_doc_path_rejects_paths_outside_docs() -> None:
    current_doc = manual.DOCS_ROOT / "user-manual/index.md"

    with pytest.raises(ValueError):
        manual._resolve_doc_path(current_doc, "../../../../etc/passwd")


def test_render_supports_anchor_navigation(tmp_path: Path) -> None:
    script_path = tmp_path / "manual_app.py"
    script_path.write_text("from apps.streamlit.views.manual import render\nrender()\n")

    at = AppTest.from_file(str(script_path), default_timeout=30)
    at.query_params["manual"] = "user-manual/monitoring.md"
    at.query_params["manual_anchor"] = "paper-trading-tab"

    at.run()

    assert at.subheader[0].value == "User Manual"
    assert at.caption[0].value == "Source: docs/user-manual/monitoring.md"
    assert at.info[0].value == "Jumped to section: Paper Trading Tab"
    assert "## Paper Trading Tab" in at.markdown[1].value


def test_app_navigation_exposes_user_manual_view() -> None:
    app_path = Path(__file__).resolve().parents[4] / "apps/streamlit/app.py"

    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.query_params["view"] = "User Manual"
    at.query_params["manual"] = "user-manual/index.md"

    at.run()

    assert "User Manual" in at.radio[0].options
    assert at.radio[0].value == "User Manual"
    assert at.subheader[0].value == "User Manual"
    assert any(
        caption.value == "Source: docs/user-manual/index.md" for caption in at.caption
    )
