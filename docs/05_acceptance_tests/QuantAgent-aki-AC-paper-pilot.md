# QuantAgent-aki — Acceptance Criteria: Paper Trading Pilot

**Issue:** QuantAgent-aki  
**Related:** [RQ](../01_requirements/QuantAgent-aki-RQ-paper-pilot.md) | [PL](../02_planning/QuantAgent-aki-PL-paper-pilot.md)

---

## How to Validate

Each criterion below is independently verifiable by the tester or Tech Lead after the implementer phase completes.

---

## AC1 — Versioned Runbook Exists

**Check:** File `docs/06_implementation/QuantAgent-aki-IM-pilot-runbook.md` exists and contains:
- A "Pilot Configuration" section with explicit values for: strategies, universe, cycles, environment, timeframe, lookback
- A "Pre-run checklist" section
- A "How to run" section with an exact shell command
- Success/failure exit criteria

**Pass condition:** File is present, committed, and contains all sections above.  
**Testable as:** `ls docs/06_implementation/QuantAgent-aki-IM-pilot-runbook.md && grep -c "Pre-run\|How to run\|exit criteria" docs/06_implementation/QuantAgent-aki-IM-pilot-runbook.md`

---

## AC2 — Pilot Script Executable

**Check:** File `scripts/run_paper_pilot.py` exists and can be imported without errors:
```bash
/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -c "import scripts.run_paper_pilot" 2>&1
# OR
/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m compileall -q scripts/run_paper_pilot.py
```

**Pass condition:** No import or syntax errors.  
**Note:** Full execution is validated in AC3/AC4.

---

## AC3 — Pilot Evidence File Produced

**Check:** After running the pilot script, a file `pilot_evidence.json` is present under `docs/envelopes/QuantAgent-aki/` with the following top-level keys:
- `pilot_id`
- `config` (dict with strategies, universe, cycles, environment)
- `cycles` (list of N dicts, each with heartbeat_status, signal_count, order_count, fill_count, error_count)
- `aggregate` (dict with totals)
- `blockers_detected` (list, empty if none)

**Pass condition:** File exists, is valid JSON, all required keys present.  
**Testable as:**
```bash
python3 -c "
import json, sys
d = json.load(open('<path>/pilot_evidence.json'))
required = ['pilot_id', 'config', 'cycles', 'aggregate', 'blockers_detected']
missing = [k for k in required if k not in d]
assert not missing, f'Missing keys: {missing}'
print('AC3 PASS')
"
```

---

## AC4 — Pilot Ran at Least 1 Complete Cycle

**Check:** `pilot_evidence.json` has `aggregate.cycles_completed >= 1`.

If `cycles_completed == 0`, the pilot is considered BLOCKED and AC4 fails. The evidence file must still exist with a `blockers_detected` list explaining why.

**Pass condition:** `aggregate.cycles_completed >= 1`  
**Note:** 0 signals or 0 trades does NOT fail AC4 — thin signal conditions are a valid outcome. What matters is that the cycle executed without crashing.

---

## AC5 — Readiness Report Produced

**Check:** File `readiness_report.md` exists under `docs/envelopes/QuantAgent-aki/` and contains:
- A "Recommendation" section with an explicit **GO**, **NO-GO**, or **CONDITIONAL GO** verdict
- A "Cycle Summary" table or equivalent
- A "Blockers Detected" section (or explicit "None detected")

**Pass condition:** File exists, recommendation verdict is present.  
**Testable as:**
```bash
grep -i "GO\|NO-GO\|CONDITIONAL" <path>/readiness_report.md | head -5
```

---

## AC6 — If NO-GO: Blockers Are Actionable

**Condition:** Only evaluated if the readiness report verdict is NO-GO or CONDITIONAL GO.

**Check:** Each blocker listed in `pilot_evidence.json` (`blockers_detected`) must have:
- A `description` field (what failed)
- An `evidence` field (traceback, log snippet, or DB query result)
- A `suggested_ticket_title` field (so Tech Lead can create a follow-up ticket)

**Pass condition:** All blockers in the list have the three fields above.

---

## AC7 — No Real Capital / Broker Interaction

**Check:** Grep the pilot script and evidence for any Alpaca/broker API calls:
```bash
grep -r "alpaca\|broker_real\|live_trading\|APCA" scripts/run_paper_pilot.py
```

**Pass condition:** Zero matches (paper broker only).

---

## Non-Acceptance Criteria (Clarifications)

- **Signals not firing** is NOT a failure — market conditions may not trigger RSI or 52-week signals in a 3-cycle window. The report must note this explicitly.
- **UI not updated** is acceptable — this pilot does not require Streamlit changes.
- **QA docker not involved** — the pilot runs against local dev DB, not QA environment.
