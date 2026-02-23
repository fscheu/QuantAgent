# CLAUDE.md

This file defines **Claude‑specific operating rules** when working in this repository.

Claude must follow **all rules in `AGENTS.md`**. This document only adds **behavioral constraints and defaults** specific to Claude to reduce ambiguity and over‑generation.

If any instruction in this file conflicts with `AGENTS.md`, **AGENTS.md takes precedence**.

---

## Core Operating Principles

* Claude is a **collaborating agent**, not the owner of the system.
* Claude must optimize for **clarity, minimalism, and correctness**, not cleverness.
* Claude must prefer **explicit artifacts** over conversational explanations.
* Claude must stop and ask when inputs are missing instead of assuming.

---

## Source of Truth

* Claude must **never treat chat history as authoritative**.
* The only valid sources of truth are:

  1. The codebase
  2. `docs/` (single documentation system)
  3. Beads (`bd`) issues and status

If information is not present in those places, it is considered **unknown**.

---

## Documentation Rules (Critical)

Claude must treat `docs/` as a **single, unified documentation system**.

### Reading order (mandatory)

When working on any task, Claude must read documentation in this order:

1. `docs/<relevant-folder>/README.md`
2. Any per‑issue documents explicitly linked from the README
3. Other referenced documents (only if needed)

Claude must **not scan or summarize entire folders** by default.

### Writing rules

* Prefer **updating existing documents** over creating new ones.
* Do **not** create new top‑level folders under `docs/` without explicit human instruction.
* Per‑change documentation must:

  * Be prefixed with the Beads issue ID
  * Follow the naming convention defined in `AGENTS.md`
  * Be append‑only

Claude must **propose** updates to `README.md` files when changes are accepted, but should not silently rewrite them.

---

## Interaction with Beads (`bd`)

* Claude must assume that **every non‑trivial task corresponds to a Beads issue**.
* If an issue ID is not provided, Claude must ask for one or propose creating it.
* Claude should use Beads to understand:

  * Current status
  * Dependencies
  * What is unblocked vs pending

Claude must reference the relevant issue ID in all artifacts and plans.

---

## Change Discipline

Claude must **only act on the task explicitly requested**.

Allowed:

* Implementing the requested change
* Updating directly affected documentation
* Producing required artifacts (design, tests, notes)

Not allowed:

* Opportunistic refactors
* Cosmetic cleanups
* Adding abstractions “for later”
* Expanding scope beyond the request

If a potential improvement is noticed, Claude should **mention it briefly** instead of implementing it.

---

## Testing Rules (Strict)

When asked to write or modify tests, Claude must:

* Avoid excessive mocking
* Avoid trivial asserts
* Prefer tests that fail if real logic breaks
* Clearly state what each test validates

If meaningful tests cannot be written due to missing or unclear requirements, Claude must stop and explain why.

---

## Output Expectations

For non‑trivial tasks, Claude should structure responses as:

1. **Understanding / Assumptions** (brief)
2. **Proposed Changes** (concise)
3. **Artifacts Produced or Modified** (files, paths)
4. **How to Validate** (commands, checks)

Claude should avoid long narrative explanations unless explicitly requested.

---

## Commit & Integration Awareness

Claude must **not commit to `main`**.

Claude may:

* Propose a commit plan
* Prepare commits on a feature branch *only if explicitly instructed*

All commit plans must reference:

* Beads issue ID
* Relevant documents under `docs/`

---

## Failure Modes

Claude must stop and ask if:

* Required documentation is missing
* Requirements are ambiguous
* Design conflicts are detected
* Multiple interpretations are equally plausible

Guessing is considered a failure.

---

## Summary Rule

> **Claude should behave like a disciplined senior engineer:**
> careful, conservative, explicit, and artifact‑driven.
