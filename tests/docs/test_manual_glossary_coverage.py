from pathlib import Path
from hashlib import sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("docs/02_planning/2026-10-28_roadmap_post_m5.md", "## 0.", "## 1."),
    ("docs/02_planning/2026-10-28_roadmap_post_m5.md", "## 5.", "## 6."),
    ("docs/02_planning/QuantAgent-lp1-PL-m2-validacion-datos-reales.md", "## 14.", None),
    ("docs/02_planning/2026-10-20_analisis_eldashboard_tableros.md", "## 2.", "## 3."),
    ("docs/02_planning/2026-10-20_analisis_eldashboard_tableros.md", "## 13.", None),
)
TERMS = (
    "Hypothesis", "Funnel", "Sharpe deflated", "Reserve", "C0", "1/N", "Shadow", "Paper", "PDT",
    "Snapshot", "Manifest", "Train, test, and reserve", "Buy and hold", "Intrabar", "Control (backtesting.py)",
    "Golden test", "Walk-forward", "Sovereign bond", "Yield to maturity (YTM, TIR)", "Duration", "Convexity",
    "Yield curve", "VWAP", "TNA and TEM", "Peso instruments", "MEP, CCL, official, and swap dollar",
    "Implied rate", "Open interest", "Implied forward", "Cointegration", "Nelson–Siegel / Diebold–Li",
    "Breakeven", "Country risk", "DV01", "Butterfly", "Real yield", "Basis point",
)
SOURCE_HASHES = (
    "b639ddfbefcaf1fbb45fddfc76539f64002c59b263965c671245ba3a56a9d8c8",
    "1b56a72d4ca3f4850f09b986b9fc298f7e1dda82f00dfb91efc828343101affd",
    "3b11187987615d537a9af0706a879029faa2bc91f664afdfafdf764096d731e9",
    "235d47ef8b360c077b01818d9b21c505624860760a5c6e213ea15c2cc69a4698",
    "b6767cde761cf095f3c0018532f6ebd59c79b742a6b4cf90493d361c6905a884",
)


def test_source_terms_have_glossary_entries():
    glossary = (ROOT / "docs/manual/glossary.md").read_text()
    for (path, start, end), expected_hash in zip(SOURCES, SOURCE_HASHES):
        source = (ROOT / path).read_text()
        section = source.split(start, 1)[1]
        if end:
            section = section.split(end, 1)[0]
        # Any source edit requires a fresh term review before updating this hash.
        assert sha256(section.encode()).hexdigest() == expected_hash
        for term in TERMS:
            if term.lower() in section.lower():
                assert f"### {term}" in glossary
    assert all(f"### {term}" in glossary for term in TERMS)
