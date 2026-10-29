# How we work

Work is grouped into **lotes** (lots): 5 to 10 related tickets that together leave something the system can do and a command that shows it (the lot's checkpoint). The queue of tickets lives in `PLAN-CONTINUACION.md` §4; ticket state lives in Beads (`bd`). Durable decisions live in `docs/04_decisions/` and are indexed in [decisions](decisions.md); see the [glossary](glossary.md) for terms.

## One delivery

1. The loop takes the first eligible row of the active lot and an agent (Codex or Gemini through `agy`) implements one small change on a `loop/<ticket>` branch.
2. The delivery is capped at 400 added lines and 10 files, with a target of 100. A ticket that does not fit is split automatically into child tickets, each with its own pull request.
3. The pull request goes to the lot branch (`lote/<name>`), never to `main`.

## Review by the PM

The PM reads the ticket before the diff, reads the whole diff, runs the check named in the ticket and confirms that CI is green on the last commit. The decision is written as a comment with an exact format:

`PM: <ticket> vi: <a concrete fact from the delivery> decido: merge | cambio <what> | descarto <why> | escalo <question>`

`merge` merges the pull request into the lot. `cambio` sends it back to the agent with the exact change. `escalo` is used for decisions that are Fede's: money or risk semantics, a changed reference number, or a second `cambio` on the same ticket. Two failed deliveries in a row stop the lot, because it means the tickets are badly defined.

## Review by Fede

When no eligible rows remain, the PM runs the lot's checkpoint, reviews this manual against what the lot changed, fills in the body of the lot pull request (including a `Manual:` line) and marks it ready. Fede tries the checkpoint and answers with:

`R: lote-<name> vi: <what was checked> decido: merge`

Only then does the lot merge into `main`. A `PM:` line is not enough to merge into `main`.

## Rules that keep the manual current

- A pull request that adds or changes a CLI command regenerates the [command reference](reference/cli/README.md) and passes its test.
- A pull request that records a new decision adds one line to the decision index and updates [criteria](criteria.md) if a number changes.
- Before a lot is marked ready, the PM reviews `docs/manual/` and reports it in the `Manual:` line.

The full contract is in `PLAN-CONTINUACION.md` §3 and `AGENTS.md`.
