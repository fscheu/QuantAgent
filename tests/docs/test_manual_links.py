import re
from pathlib import Path


def test_manual_readme_links_resolve():
    readme = Path("docs/manual/README.md")
    links = re.findall(r"\]\(([^)#]+(?:\.md))\)", readme.read_text())
    assert links
    assert {Path(link).name for link in links} >= {
        "overview.md", "operations.md", "how-we-work.md", "criteria.md",
        "decisions.md", "glossary.md", "funnel.md", "README.md",
    }
    assert all((readme.parent / link).is_file() for link in links)
