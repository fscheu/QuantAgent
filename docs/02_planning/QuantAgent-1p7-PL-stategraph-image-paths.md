# QuantAgent-1p7 — Planning — StateGraph Image Paths

## Objective
Make StateGraph visualization artifacts disk-backed and path-referenced with the smallest possible code change.

## Dependencies
- Existing graph visualization method on the compiled LangGraph object.
- Existing local artifact conventions (`data/artifacts/...`).
- No schema change assumed.

## Task Breakdown

1. **Trace the current visualization entry point**
   - Identify where the compiled graph image is currently generated/returned.
   - Confirm which consumer currently expects the in-memory payload.

2. **Add a minimal export helper**
   - Implement export in `quantagent/trading_graph.py` (preferred) or the closest orchestration layer.
   - Write bytes to a PNG under the local artifact tree.
   - Return a path string.

3. **Replace payload references with path references**
   - Update the immediate consumer(s) to propagate `stategraph_image_path` instead of image bytes/base64.
   - Keep the change local; do not refactor unrelated artifact flows.

4. **Add focused tests**
   - Mock the visualization byte producer.
   - Assert file creation under a temp directory.
   - Assert path-only metadata/reference behavior.
   - Assert artifacts policy `none` skips file creation.

5. **Verification + handoff**
   - Run targeted tests.
   - Run `ruff check --fix .`.
   - Run `python -m compileall -q quantagent tests`.

## Suggested Files
- `quantagent/trading_graph.py`
- Possibly one immediate consumer of visualization metadata
- `tests/` file covering the export helper / propagation path

## Risks
- LangGraph renderer availability can vary by environment; tests should mock the byte-producing method instead of depending on actual rendering.
- The existing consumer may be implicit rather than centralized; if so, keep the first implementation narrowly scoped and document any uncovered follow-up.

## Recommended Next Step
Route to **autodev-implementer** on the feature branch for a minimal code change plus targeted tests.
