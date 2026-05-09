# QuantAgent-1p7 — Requirements — StateGraph Image Paths

## Objective
Persist LangGraph `StateGraph` visualizations as image files on disk and propagate file paths instead of in-memory image payloads.

## Detail Level
STANDARD

## Assumptions
- The existing graph visualization path already produces image bytes for the compiled graph; this ticket changes where that artifact is stored and how it is referenced.
- "Reference file paths" means downstream metadata, snapshots, or UI/debug surfaces should carry a filesystem path string, not raw image bytes/base64.
- No new storage service is required; local disk is sufficient for the current test/non-production app.

## Scope In
- Exporting the compiled LangGraph / StateGraph visualization to disk.
- Returning or recording the exported image path in the existing metadata/reference flow.
- Aligning the graph visualization flow with the repo-wide `path-only` artifact policy.
- Minimal tests proving disk persistence and path-only references.

## Scope Out
- Rewriting graph topology, agents, or backtesting logic.
- Changing chart generation for candlestick/pattern/trend artifacts beyond keeping conventions consistent.
- Adding object storage, CDN delivery, or artifact retention jobs.
- UI redesign beyond showing or preserving the resulting path reference.

## Functional Requirements

### FR1 — Disk-backed export
When graph visualization is requested, QuantAgent must write the generated StateGraph image to disk and produce a path reference to that file.

### FR2 — Path-only references
Long-lived structures must reference the StateGraph image by file path only. The implementation must not keep the rendered image bytes/base64 inside persisted state, checkpoints, config snapshots, or other durable metadata.

### FR3 — Predictable artifact location
The saved image path must live under the same local artifact conventions already used by the project for path-based images so that debugging and manual inspection are straightforward.

### FR4 — Existing policy compatibility
If the caller/run is configured for `path-only`, the graph visualization flow must obey that mode. If artifacts are disabled (`none`), the flow must skip creating the file and must not fabricate a path.

### FR5 — Reference propagation
Any existing surface that currently exposes the graph visualization result must expose the file path instead of an in-memory payload.

## Constraints
- Keep the change minimal and local to graph visualization / artifact reference plumbing.
- Reuse existing artifact conventions (`data/artifacts/...`) rather than inventing a second storage layout.
- Do not require a schema migration unless a concrete missing field blocks propagation.

## Edge Cases
- Multiple runs for the same symbol/thread must not silently overwrite each other; filenames should include enough context to stay distinct.
- Relative vs absolute path handling must be consistent for the same call path.
- Tests must not depend on real LLM calls or a real renderer service.

## Definition of Done
- StateGraph visualization is written to disk when enabled.
- The code path returns/stores a file path reference, not raw image bytes.
- Artifact-disabled runs skip generation cleanly.
- Tests cover the export and reference behavior.
