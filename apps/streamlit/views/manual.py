from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import streamlit as st

DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs"
DEFAULT_DOC = Path("user-manual/index.md")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class ResolvedDocument:
    path: Path
    anchor: str | None = None


def _slugify_heading(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"[-\s]+", "-", slug).strip("-")


def _split_target(target: str) -> tuple[str, str | None]:
    base, _, anchor = target.partition("#")
    return base, anchor or None


def _resolve_doc_path(current_doc: Path, target: str) -> ResolvedDocument:
    base_target, anchor = _split_target(target.strip())
    if not base_target:
        return ResolvedDocument(current_doc, anchor)

    candidate = (current_doc.parent / base_target).resolve()
    docs_root = DOCS_ROOT.resolve()
    try:
        candidate.relative_to(docs_root)
    except ValueError as exc:
        raise ValueError(f"Unsupported path outside docs/: {target}") from exc

    if candidate.suffix.lower() != ".md":
        raise ValueError(f"Only markdown files are supported: {target}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Linked document not found: {target}")
    return ResolvedDocument(candidate, anchor)


def _doc_param_from_path(path: Path) -> str:
    return path.resolve().relative_to(DOCS_ROOT.resolve()).as_posix()


def _build_manual_href(target: ResolvedDocument) -> str:
    params = [
        f"view={quote('User Manual')}",
        f"manual={quote(_doc_param_from_path(target.path))}",
    ]
    if target.anchor:
        params.append(f"manual_anchor={quote(target.anchor)}")
    return f"?{'&'.join(params)}"


def _rewrite_markdown_links(markdown_text: str, current_doc: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:")):
            return match.group(0)
        if not target.startswith("#") and not target.lower().endswith(".md") and ".md#" not in target.lower():
            return match.group(0)
        try:
            resolved = _resolve_doc_path(current_doc, target)
        except (FileNotFoundError, ValueError):
            return f"{label} (`{target}` unavailable in app)"
        return f"[{label}]({_build_manual_href(resolved)})"

    return MARKDOWN_LINK_RE.sub(replace, markdown_text)


def _extract_headings(markdown_text: str) -> list[tuple[int, str, str]]:
    headings: list[tuple[int, str, str]] = []
    for match in HEADING_RE.finditer(markdown_text):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append((level, title, _slugify_heading(title)))
    return headings


def _extract_anchor_section(markdown_text: str, anchor: str) -> tuple[str, str] | None:
    lines = markdown_text.splitlines()
    matches: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not match:
            continue
        title = match.group(2).strip()
        matches.append((idx, len(match.group(1)), title))

    for pos, (start_idx, level, title) in enumerate(matches):
        if _slugify_heading(title) != anchor:
            continue
        end_idx = len(lines)
        for next_idx, next_level, _ in matches[pos + 1 :]:
            if next_level <= level:
                end_idx = next_idx
                break
        section = "\n".join(lines[start_idx:end_idx]).strip()
        return title, section
    return None


def _get_requested_doc() -> tuple[Path, str | None]:
    requested = st.query_params.get("manual", DEFAULT_DOC.as_posix())
    anchor = st.query_params.get("manual_anchor")
    candidate = (DOCS_ROOT / requested).resolve()
    docs_root = DOCS_ROOT.resolve()
    try:
        candidate.relative_to(docs_root)
    except ValueError as exc:
        raise ValueError(f"Unsupported manual path: {requested}") from exc
    return candidate, anchor


def _render_navigation(current_doc: Path, headings: list[tuple[int, str, str]]) -> None:
    manual_docs = sorted((DOCS_ROOT / "user-manual").glob("*.md"))
    doc_options = {_doc_param_from_path(path): path.stem.replace("-", " ").title() for path in manual_docs}
    current_param = _doc_param_from_path(current_doc)
    if current_param not in doc_options:
        doc_options[current_param] = current_param

    selected_doc = st.selectbox(
        "Manual page",
        list(doc_options.keys()),
        index=list(doc_options.keys()).index(current_param),
        format_func=doc_options.get,
    )
    if selected_doc != current_param:
        st.query_params["view"] = "User Manual"
        st.query_params["manual"] = selected_doc
        st.query_params.pop("manual_anchor", None)
        st.rerun()

    if headings:
        anchor_options = [""] + [anchor for _, _, anchor in headings]
        anchor_labels = {"": "Top of document"}
        anchor_labels.update({anchor: title for _, title, anchor in headings})
        current_anchor = st.query_params.get("manual_anchor", "")
        selected_anchor = st.selectbox(
            "Section",
            anchor_options,
            index=anchor_options.index(current_anchor) if current_anchor in anchor_options else 0,
            format_func=anchor_labels.get,
        )
        if selected_anchor != current_anchor:
            st.query_params["view"] = "User Manual"
            st.query_params["manual"] = current_param
            if selected_anchor:
                st.query_params["manual_anchor"] = selected_anchor
            else:
                st.query_params.pop("manual_anchor", None)
            st.rerun()


def render() -> None:
    st.subheader("User Manual")

    try:
        current_doc, current_anchor = _get_requested_doc()
    except ValueError as exc:
        st.error(str(exc))
        return

    if current_doc.suffix.lower() != ".md":
        st.error(f"Only markdown documents are supported: {current_doc.name}")
        return
    if not current_doc.is_file():
        st.error(f"Manual page not found: {current_doc.relative_to(DOCS_ROOT)}")
        return

    markdown_text = current_doc.read_text(encoding="utf-8")
    headings = _extract_headings(markdown_text)
    _render_navigation(current_doc, headings)

    st.caption(f"Source: docs/{_doc_param_from_path(current_doc)}")
    st.markdown(f"[Open manual home]({_build_manual_href(ResolvedDocument(DOCS_ROOT / DEFAULT_DOC))})")

    if current_anchor:
        anchor_section = _extract_anchor_section(markdown_text, current_anchor)
        if anchor_section:
            title, section = anchor_section
            st.info(f"Jumped to section: {title}")
            st.markdown(_rewrite_markdown_links(section, current_doc))
            st.divider()
        else:
            st.warning(f"Section not found in this document: #{current_anchor}")

    st.markdown(_rewrite_markdown_links(markdown_text, current_doc))
