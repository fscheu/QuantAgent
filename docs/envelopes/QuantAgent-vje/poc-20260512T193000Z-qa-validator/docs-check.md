# Docs Consistency Check — QuantAgent-vje
**Issue:** QuantAgent-vje — Scheduler status and controls in Streamlit Dashboard  
**Run ID:** poc-20260512T193000Z-qa-validator  
**Date:** 2026-05-12

---

## Docs Sources Reviewed

1. **Requirements:** `docs/01_requirements/QuantAgent-vje-RQ-scheduler-monitoring-dashboard.md`
2. **Acceptance Criteria:** `docs/05_acceptance_tests/QuantAgent-vje-AC-scheduler-monitoring-dashboard.md`
3. **User Manual:** `docs/user-manual/dashboard.md`

---

## Expected User-Facing Behavior (from docs)

### Core Feature Description

**From Requirements (FR-3):**
- New "Paper Trading" tab in Streamlit dashboard
- Status card showing:
  - Current status: 🟢 Active / 🟡 Stale / 🔴 Stopped
  - Last run timestamp (humanized, e.g., "2 minutes ago")
  - Cycle stats when available
- Recent runs table: last 10 scheduler cycles
- Graceful handling when table doesn't exist or no data available

**From Acceptance Criteria:**
- **AC-5:** "Paper Trading" tab visible in tab list
- **AC-6:** Status card shows 🟢 Active when scheduler ran <2h ago
- **AC-7:** Status card shows 🟡 Stale when 2-24h ago
- **AC-8:** Status card shows 🔴 Stopped when >24h or no heartbeat
- **AC-9:** Recent runs table displays up to 10 rows
- **AC-10:** Graceful handling when `scheduler_heartbeats` table missing

**From User Manual:**
- Section "Paper Trading Scheduler Status" describes:
  - Daily automation check workflow
  - Status indicators and what they mean
  - Instructions on what to do when status is Stale or Stopped
  - Related links to Paper Trading Automation and Monitoring guides

### Out of Scope (confirmed)

- Starting/stopping scheduler from UI
- Real-time streaming updates (polling sufficient)
- Historical charts/metrics
- Scheduler configuration editing from UI

---

## Expected Visual Elements

### Primary Elements (must be present)

1. **Tab:** "Paper Trading" visible in main tab list
2. **Heading:** "📊 Paper Trading Scheduler" (heading level 2)
3. **Status card** (when data available):
   - Status emoji + text (Active/Stale/Stopped)
   - Last run timestamp
   - Assets processed count
   - Duration
4. **Recent runs table** (when data available):
   - Columns: Timestamp, Duration, Assets, Errors, Status
   - Up to 10 rows

### Fallback/Degraded Mode (when no data)

**From AC-10 and observed implementation:**
- Warning message: "⚠️ No scheduler heartbeat found"
- Info message: "The scheduler may not be running, or no cycles have completed yet for environment: **{environment}**"
- Instructions on how to start scheduler: `python apps/paper_trading.py`
- Should NOT crash or show stack traces

---

## Expected Behavior in Current Environment

**Current state:**
- Database partially initialized (some tables missing)
- Scheduler not running (no heartbeat data)
- Expected: graceful degradation

**Docs consistency:**
- Requirements explicitly cover graceful handling (FR-3, AC-10)
- User manual references monitoring workflow assuming data exists
- Implementation should show fallback UI, not error

---

## Validation Scope for PoC

### In Scope (minimal viable validation)

1. Tab "Paper Trading" exists and is clickable
2. Heading "📊 Paper Trading Scheduler" renders
3. Fallback message displays when no heartbeat available
4. No console errors that break the page
5. No stack traces visible in UI

### Out of Scope (for PoC; would require scheduler running)

- Status card with real data (Active/Stale/Stopped logic)
- Recent runs table populated
- Duration calculations
- Assets processed counts
- Link to last trade

---

## Docs Consistency Assessment (preliminary)

**Status:** DOCS_OK (for PoC scope)

**Reasoning:**
- Requirements, acceptance criteria, and user manual are aligned on core feature
- Graceful degradation explicitly documented (AC-10)
- Current environment limitations are understood
- No contradictions found between requirement and acceptance docs

**Note for full validation:**
- Full validation would require scheduler running with heartbeat data
- User manual describes rich functionality (status thresholds, recent runs table)
- Those elements cannot be validated without live data
- This PoC focuses on structural presence and graceful no-data handling

---

## Next Steps

Proceed to browser validation:
1. Navigate to http://127.0.0.1:8501
2. Verify "Paper Trading" tab exists
3. Click tab and confirm heading renders
4. Confirm fallback message displays
5. Check browser console for errors
6. Capture screenshots

---
