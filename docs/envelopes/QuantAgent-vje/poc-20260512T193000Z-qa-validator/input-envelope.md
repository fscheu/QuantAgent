---
run_id: poc-20260512T193000Z-qa-validator
phase: qa_validator
mode: manual-poc
issue_id: QuantAgent-vje
title: Scheduler status and controls in Streamlit Dashboard
executor: hermes-internal

target_url: http://127.0.0.1:8501
deployed_url: http://127.0.0.1:8501

docs_paths:
  - docs/01_requirements/QuantAgent-vje-RQ-scheduler-monitoring-dashboard.md
  - docs/05_acceptance_tests/QuantAgent-vje-AC-scheduler-monitoring-dashboard.md
  - docs/user-manual/dashboard.md

expected_checks:
  - type: tab_visible
    value: "Paper Trading"
  - type: heading_present
    value: "📊 Paper Trading Scheduler"
  - type: graceful_no_data
    description: "UI should show fallback message when no heartbeat exists, not crash"
  - type: console_errors_max
    value: 0
    description: "No critical console errors when viewing Paper Trading tab"

context: |
  QuantAgent-vje adds a "Paper Trading" tab to the Streamlit dashboard to monitor the TradingScheduler status.
  
  Core features:
  - New "Paper Trading" tab visible in the UI
  - Status card showing scheduler health (Active/Stale/Stopped)
  - Recent runs table (last 10 scheduler cycles)
  - Graceful fallback when no heartbeat data exists
  
  Out of scope for this ticket:
  - Starting/stopping scheduler from UI
  - Real-time streaming updates
  - Historical charts
  
  Current environment limitation:
  - Database is not fully initialized (trades/strategy_configs tables missing)
  - Scheduler is not currently running (no heartbeat data)
  - Expected behavior: UI should show graceful fallback, not crash
  
  Success criteria for PoC:
  - Tab exists and is navigable
  - Heading/title renders correctly
  - Fallback message displays when no data available
  - No console errors that break the page

---
