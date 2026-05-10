# QuantAgent-1p7 — Acceptance Criteria — StateGraph Image Paths

## AC1 — Export writes a file
**Given** a compiled trading graph and artifact saving enabled  
**When** the StateGraph visualization export is requested  
**Then** QuantAgent writes a PNG file to disk and returns or records the created path.

## AC2 — Path-only persistence
**Given** a run that exports the StateGraph visualization  
**When** QuantAgent persists or exposes visualization metadata  
**Then** the reference is a file path string  
**And** the persisted/exposed metadata does not contain raw PNG bytes or base64 image content.

## AC3 — Artifact policy respected
**Given** a run configured with artifacts policy `none`  
**When** the graph visualization flow executes  
**Then** QuantAgent does not write the StateGraph image to disk  
**And** no fake placeholder path is stored.

## AC4 — Layout follows local artifact conventions
**Given** a run with available execution context (such as environment, run/thread, or symbol)  
**When** the image is exported  
**Then** the resulting path is created under the repo's local artifact layout rather than an ad-hoc temporary location.

## AC5 — Deterministic automated proof
**Given** the automated test suite for this ticket  
**When** the relevant tests are run  
**Then** they prove file creation and path-only references without requiring live LLM calls or a real external renderer service.
