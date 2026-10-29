# How we work

Work is grouped into lots. The PM selects an eligible ticket, assigns one small implementation, and reviews its pull request against the lot branch. Each delivery states what changed, what to inspect, how to reproduce it, and any risk. The loop contract and queue live in [PLAN-CONTINUACION.md](../../PLAN-CONTINUACION.md).

The PM checks scope, diff size, acceptance evidence, and CI, then integrates accepted work into the lot. Fede reviews the complete lot and decides whether it is ready to merge to `main`. `main` is protected; individual loop deliveries do not merge directly to it.

Tickets and dependencies are tracked in Beads; durable requirements, decisions, and technical context live under `docs/`. A delivery changes only the ticket's requested files and leaves unrelated work for its own ticket.
