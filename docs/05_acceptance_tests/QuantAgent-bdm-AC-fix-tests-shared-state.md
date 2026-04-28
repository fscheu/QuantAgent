# QuantAgent-bdm — Acceptance Criteria — Fix tests expecting shared state updates across multi-agent calls

## A. Analysis agents do not update `messages`

### A1 — Sequential calls: no `messages` mutation
- Given an initial state without `messages` (or with `messages=[]`)
- When the Indicator Agent is invoked with that state
- Then the returned update contains `indicator_report`
- And the state’s `messages` key is unchanged (still absent or still the same list)

- Given the state after applying the Indicator Agent update
- When the Pattern Agent is invoked
- Then the returned update contains `pattern_report`
- And the state’s `messages` key is unchanged

- Given the state after applying the Pattern Agent update
- When the Trend Agent is invoked
- Then the returned update contains `trend_report`
- And the state’s `messages` key is unchanged

### A2 — No test asserts “all agents produce messages”
- Given the test suite
- When searching for assertions that require analysis agents to return/update `messages`
- Then no such assertions remain

## B. Decision agent is the only writer to `messages`

### B1 — Decision invocation adds/returns `messages`
- Given a state that contains `indicator_report`, `pattern_report`, and `trend_report`
- And a baseline `messages` value (absent or a list)
- When the Decision Agent is invoked
- Then the returned update includes `final_trade_decision`
- And the resulting state contains `messages` as a list

## C. Full compiled graph behavior

### C1 — Compiled graph returns `messages` because Decision Agent ran
- Given a valid initial state for the compiled trading graph
- When the compiled graph is invoked end-to-end
- Then the result includes `indicator_report`, `pattern_report`, `trend_report`, and `final_trade_decision`
- And the result includes `messages` as a list
- And no test requires intermediate analysis nodes to have updated `messages`
