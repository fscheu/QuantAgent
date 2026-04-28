# QuantAgent-bdm — Plan — Fix tests expecting shared state updates across multi-agent calls

## Level of Detail
MINIMAL (bugfix constrained to test expectations; existing design doc already defines the state contract).

## Execution Plan (0.5–2h tasks)
1. Identify failing tests and incorrect assertions
   - Locate tests that assert analysis agents update `messages` or that “all agents produce messages”.
   - Confirm current production behavior against `docs/03_design/MESSAGE_STATE_MANAGEMENT.md`.

2. Update multi-agent/sequential tests
   - Replace assertions that depend on `messages` being updated by analysis agents.
   - Prefer assertions on presence/type of structured report keys.
   - Where needed, assert `messages` is unchanged across analysis-agent calls.

3. Update full-graph tests
   - Ensure end-to-end compiled graph tests assert `messages` presence only as a Decision Agent outcome.
   - Ensure required keys include structured reports + decision output; avoid implying intermediate message updates.

4. Run test suite (local/CI parity)
   - Run the smallest relevant subset first, then full suite:
     - `pytest -q` (or targeted files if CI is slow)

## Risks / Gotchas
- Some tests may implicitly depend on `messages` being initialized in fixtures; align fixtures to the contract (analysis agents shouldn’t require `messages`).
- Avoid replacing incorrect assertions with tautologies; keep checks anchored to the documented contract.

## Handoff Notes
- If a mismatch is discovered between tests and production behavior, treat `docs/03_design/MESSAGE_STATE_MANAGEMENT.md` as the intended contract and adjust tests accordingly (do not change production code in this planning-only step).
