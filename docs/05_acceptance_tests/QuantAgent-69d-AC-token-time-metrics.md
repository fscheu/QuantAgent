# Acceptance Criteria: Track LLM token usage and runtime metrics

**Issue:** QuantAgent-69d  
**Related:** [RQ](../01_requirements/QuantAgent-69d-RQ-token-time-metrics.md) | [DS](../03_design/QuantAgent-69d-DS-token-time-metrics.md)

## AC-1: Successful call persists telemetry
**Given** an instrumented LLM call completes successfully  
**When** the call returns  
**Then** one `logs` row with `event_type = "llm_call"` exists  
**And** it contains `duration_ms > 0`  
**And** includes `provider`, `model`, and `operation` in structured fields/metadata

## AC-2: Token fields are nullable by provider capability
**Given** a provider response does not expose token usage  
**When** the telemetry row is persisted  
**Then** token fields are null  
**And** `duration_ms` plus execution context are still present

## AC-3: Failed calls still produce evidence
**Given** an instrumented LLM call raises an exception  
**When** QuantAgent records the failure  
**Then** one `logs` row with `event_type = "llm_call"` still exists  
**And** the structured metadata marks the status as error  
**And** `duration_ms > 0`

## AC-4: Backtest aggregation stays isolated
**Given** backtest run A and backtest run B both generate LLM calls  
**When** metrics are queried for backtest run A  
**Then** the aggregate excludes rows from backtest run B

## AC-5: Session aggregation stays isolated
**Given** two executions use different `thread_id` values  
**When** metrics are queried for one `thread_id`  
**Then** only rows for that `thread_id` are included

## AC-6: Aggregate output is decision-useful
**Given** at least one stored telemetry row for a session or backtest  
**When** the aggregate query runs  
**Then** it returns call count  
**And** token sums  
**And** duration sum and average  
**And** a per-`operation` breakdown
