# Inventario de ramas remotas y worktrees (QuantAgent-cv1)

Fecha: 2026-10-08. Ramas: `git fetch --prune --unshallow origin` + `git branch -r`; fusión contra `origin/main` (3a15fc22) con `git branch -r --merged origin/main`; estados de `.beads/issues.jsonl` de `lote/engine-y-cierre-m1`.
Worktrees: listado de la VM del 2026-10-08 copiado tal cual (no visible desde este entorno). Solo inventario: no se borró ni renombró nada.

## 1. Ramas remotas

Total: 82 ramas (sin `origin/HEAD`).

| Rama | Ticket | Estado ticket | Último commit | Contra origin/main | Commits ausentes en main |
|---|---|---|---|---|---|
| `chore/ci-disable-qa-validator` | - | sin ticket | 431c87ef 2026-09-29 | merged | - |
| `chore/loop-fase0` | - | sin ticket | b9b5d0ac 2026-09-29 | merged | - |
| `chore/loop-hermes-entry` | - | sin ticket | 8a2d1f3c 2026-10-03 | merged | - |
| `feat/Core-State-Mgmnt` | - | sin ticket | c81ff48a 2025-11-27 | merged | - |
| `feat/LangGraph-Improve` | - | sin ticket | c82e46c9 2025-11-26 | merged | - |
| `feat/LangGraph-Parallel` | - | sin ticket | 368f8b16 2025-11-30 | merged | - |
| `feat/Project-Setup` | - | sin ticket | 5812e215 2025-11-25 | merged | - |
| `feat/Streamlit-App` | - | sin ticket | 619d09ce 2025-11-30 | merged | - |
| `feature/QuantAgent-046-hotfix-ci-failure-in-ci-commit-d46effc` | QuantAgent-046 | closed | 4bf6ac80 2026-02-24 | merged | - |
| `feature/QuantAgent-0b5-integrate-positionmonitor-into-tradingsc` | QuantAgent-0b5 | closed | 782a3f9e 2026-04-27 | merged | - |
| `feature/QuantAgent-0tk-ci-alembic-migrations` | QuantAgent-0tk | closed | 90d42799 2026-05-30 | merged | - |
| `feature/QuantAgent-19f-manual-viewer-salvage` | QuantAgent-19f | closed | 702c3ee2 2026-05-25 | merged | - |
| `feature/QuantAgent-1p7-save-stategraph-images-to-disk` | QuantAgent-1p7 | closed | f8c24554 2026-05-10 | merged | - |
| `feature/QuantAgent-1p7-save-stategraph-images-to-disk-and-refer` | QuantAgent-1p7 | closed | 80f4f0c8 2026-02-18 | merged | - |
| `feature/QuantAgent-339-qa-validator-runtime-real` | QuantAgent-339 | closed | 31a35423 2026-05-14 | merged | - |
| `feature/QuantAgent-375-scope-replay-signal-lookup-to-selected-s` | QuantAgent-375 | closed | ec704bad 2026-05-11 | merged | - |
| `feature/QuantAgent-3hs-fix-tradinggraph-reload-mock` | QuantAgent-3hs | closed | 3dc1d912 2026-05-07 | no mergeada | 1 |
| `feature/QuantAgent-3o4-implement-tradingscheduler-for-automatic` | QuantAgent-3o4 | closed | 2dc113dd 2026-02-25 | merged | - |
| `feature/QuantAgent-3o8-implement-replay-execution-mode-reuse-an` | QuantAgent-3o8 | closed | 7886ad2b 2026-05-10 | no mergeada | 2 |
| `feature/QuantAgent-3o8-replay-fresh-20260511T023846Z` | QuantAgent-3o8 | closed | 27dec03c 2026-05-11 | no mergeada | 1 |
| `feature/QuantAgent-3o8-replay-refresh-20260511T1328Z` | QuantAgent-3o8 | closed | 5fa31589 2026-05-11 | merged | - |
| `feature/QuantAgent-40j-fix-missing-benchmark-fixture` | QuantAgent-40j | closed | 49cdfdd4 2026-05-08 | merged | - |
| `feature/QuantAgent-4fm-externalize-config-fresh` | QuantAgent-4fm | closed | 3c298035 2026-05-10 | merged | - |
| `feature/QuantAgent-4fm-externalize-hardcoded-trading-configurat` | QuantAgent-4fm | closed | e4b4ce4d 2026-02-17 | no mergeada | 1 |
| `feature/QuantAgent-4w4-lookback-windows` | QuantAgent-4w4 | closed | 1b242877 2026-05-06 | merged | - |
| `feature/QuantAgent-577-macro-indicadores` | QuantAgent-577 | open | c29af63a 2026-10-07 | no mergeada | 1 |
| `feature/QuantAgent-69d-implementar-tracking-de-tokens-y-tiempo` | QuantAgent-69d | closed | c9c7bfd2 2026-02-18 | no mergeada | 1 |
| `feature/QuantAgent-69d-token-time-metrics-refresh` | QuantAgent-69d | closed | ce19c469 2026-05-12 | merged | - |
| `feature/QuantAgent-6t4-structured-output-agents` | QuantAgent-6t4 | closed | 4dc24116 2026-05-08 | merged | - |
| `feature/QuantAgent-6t4-use-with-structured-output-in-pattern-ag` | QuantAgent-6t4 | closed | 3bc96a0a 2026-02-18 | no mergeada | 1 |
| `feature/QuantAgent-7bn` | QuantAgent-7bn | no encontrado | 2bad83fd 2026-01-06 | merged | - |
| `feature/QuantAgent-82t-ci-tests-clean` | QuantAgent-82t | closed | ed6bb0a8 2026-05-05 | no mergeada | 3 |
| `feature/QuantAgent-82t-re-enable-unit-tests-in-ci-pipeline-main` | QuantAgent-82t | closed | 0c2a4e1f 2026-04-21 | no mergeada | 1 |
| `feature/QuantAgent-88h-create-seed-data-script-for-dev-and-qa-d` | QuantAgent-88h | closed | e80f3781 2026-05-04 | merged | - |
| `feature/QuantAgent-8yr-fix-pytest-collection-blockers` | QuantAgent-8yr | closed | 16a123a9 2026-05-07 | merged | - |
| `feature/QuantAgent-9t5-fix-stale-worktree-path-assumptions-in-wa` | QuantAgent-9t5 | closed | c4b33181 2026-05-09 | merged | - |
| `feature/QuantAgent-ait-static-util-pandas-freq` | QuantAgent-ait | closed | 340eb774 2026-05-09 | merged | - |
| `feature/QuantAgent-aki-ejecutar-piloto-controlado-de-paper-trad` | QuantAgent-aki | closed | 16d12434 2026-05-14 | merged | - |
| `feature/QuantAgent-bug-stop-tracking-repo-local-venv` | QuantAgent-bug | closed | 5779fd88 2026-05-07 | merged | - |
| `feature/QuantAgent-c69-m1-llm-agent-strategy-impl` | QuantAgent-c69 | closed | e6a42c54 2026-05-05 | merged | - |
| `feature/QuantAgent-c69-m1-llm-agent-strategy-planning` | QuantAgent-c69 | closed | 55e3aef4 2026-05-05 | merged | - |
| `feature/QuantAgent-e4k-refactor-backtest-to-depend-only-on-orde` | QuantAgent-e4k | closed | 781c789c 2026-05-12 | no mergeada | 3 |
| `feature/QuantAgent-fg0-test-revision` | QuantAgent-fg0 | closed | a35dcce6 2026-01-07 | merged | - |
| `feature/QuantAgent-g3c-position-reversal` | QuantAgent-g3c | closed | 1aa0596a 2026-01-05 | merged | - |
| `feature/QuantAgent-g3c-position-reversal-fix` | QuantAgent-g3c | closed | 5d2d24bc 2026-01-04 | merged | - |
| `feature/QuantAgent-h7d-fix-message` | QuantAgent-h7d | closed | 0414f163 2026-01-10 | merged | - |
| `feature/QuantAgent-hdx-hotfix-ci-failure-in-unit-tests-commit-6` | QuantAgent-hdx | closed | 0930b4df 2026-02-23 | no mergeada | 1 |
| `feature/QuantAgent-ia2-complete-universe-management-in-configur` | QuantAgent-ia2 | closed | cb657f9e 2026-02-25 | merged | - |
| `feature/QuantAgent-kkj.1-fix-backtest-run-dup-20260525T173935Z` | QuantAgent-kkj.1 | closed | f3c40717 2026-05-25 | merged | - |
| `feature/QuantAgent-kkj.11-configurar-routing-multi-provider-por-ro` | QuantAgent-kkj.11 | in_progress | 945c2e6b 2026-05-27 | no mergeada | 1 |
| `feature/QuantAgent-kkj.11-routing-fresh-main` | QuantAgent-kkj.11 | in_progress | 94881032 2026-05-28 | no mergeada | 2 |
| `feature/QuantAgent-kkj.11-routing-refresh-20260528` | QuantAgent-kkj.11 | in_progress | f3c0662f 2026-05-28 | no mergeada | 3 |
| `feature/QuantAgent-kkj.2-agregar-controles-de-scheduler-paper-tra` | QuantAgent-kkj.2 | closed | 794263da 2026-05-26 | merged | - |
| `feature/QuantAgent-kkj.3-ux-redise-ar-dashboard-para-ser-environm` | QuantAgent-kkj.3 | closed | 96b5d9f0 2026-05-28 | merged | - |
| `feature/QuantAgent-kkj.4-ux-separar-pesta-a-configuration-en-llm` | QuantAgent-kkj.4 | closed | dc4377b5 2026-05-28 | merged | - |
| `feature/QuantAgent-kkj.5-ux-agregar-help-contextual-en-configurat` | QuantAgent-kkj.5 | closed | ffb2c137 2026-05-29 | merged | - |
| `feature/QuantAgent-kkj.8-crear-strategy-registry-y-parametrizar-s` | QuantAgent-kkj.8 | closed | a4ff363e 2026-05-26 | merged | - |
| `feature/QuantAgent-kkj.9-agregar-selector-de-estrategia-en-ui-bac` | QuantAgent-kkj.9 | closed | 370aff54 2026-05-27 | merged | - |
| `feature/QuantAgent-l8r-fix-trade-pnl-ci-regression` | QuantAgent-l8r | closed | be11db04 2026-05-08 | merged | - |
| `feature/QuantAgent-les-support-commissions-in-p-l-calculation-clean` | QuantAgent-les | closed | 5df43769 2026-05-04 | no mergeada | 6 |
| `feature/QuantAgent-ng1-remove-redundant-main-ci-yml-workflow` | QuantAgent-ng1 | closed | 9a2e3d6b 2026-04-30 | no mergeada | 2 |
| `feature/QuantAgent-nrt-fix-backtest-position-monitor-gate-failures` | QuantAgent-nrt | closed | 48336b65 2026-05-07 | merged | - |
| `feature/QuantAgent-o2b-fix-azure-provider-gate-failures` | QuantAgent-o2b | closed | 2da0c078 2026-05-07 | merged | - |
| `feature/QuantAgent-ou3-spx-spy-mapping` | QuantAgent-ou3 | closed | fb43fc0d 2026-01-09 | merged | - |
| `feature/QuantAgent-s62-extender-observabilidad-operativa-m-nima` | QuantAgent-s62 | closed | 33a92092 2026-05-13 | merged | - |
| `feature/QuantAgent-sft-paper-runtime-hardening` | QuantAgent-sft | closed | cdc53077 2026-05-13 | no mergeada | 1 |
| `feature/QuantAgent-sft-paper-runtime-hardening-refresh-20260513T175506Z` | QuantAgent-sft | closed | 0d5dfb60 2026-05-13 | merged | - |
| `feature/QuantAgent-um8-implementar-batch-processing-para-llamad` | QuantAgent-um8 | closed | bb74cc26 2026-05-30 | merged | - |
| `feature/QuantAgent-uzq-fix-tradingscheduler-heartbeat-and-sched` | QuantAgent-uzq | closed | 833f0a95 2026-05-09 | merged | - |
| `feature/QuantAgent-vje-scheduler-status-and-controls-in-streaml` | QuantAgent-vje | closed | b6929a34 2026-04-29 | merged | - |
| `feature/QuantAgent-vna-m1-strategy-1-triple-screen-strategy-ale` | QuantAgent-vna | closed | 5004af32 2026-05-05 | no mergeada | 1 |
| `feature/QuantAgent-vna-triple-screen-strategy-alexander-elder` | QuantAgent-vna | closed | 605702ea 2026-02-18 | no mergeada | 1 |
| `feature/QuantAgent-x8u-hotfix-ci-failure-in-ci-commit-f8cf1e5` | QuantAgent-x8u | closed | 68be2bea 2026-02-24 | no mergeada | 2 |
| `feature/plan30` | - | sin ticket | 6b92013a 2026-09-26 | merged | - |
| `feature/quantagent-m1-strategy-planning-20260505` | - | sin ticket | d31f797b 2026-05-05 | merged | - |
| `gh-pages` | - | sin ticket | f0989be4 2026-02-26 | no mergeada | 21 |
| `integration/QuantAgent-82t-20260508T213950Z` | QuantAgent-82t | closed | cbaf7279 2026-05-08 | no mergeada | 3 |
| `loop/QuantAgent-832.1` | QuantAgent-832.1 | open | fe58a465 2026-10-08 | no mergeada | 2 |
| `loop/QuantAgent-bv8` | QuantAgent-bv8 | closed | 243cf5ea 2026-10-03 | merged | - |
| `loop/QuantAgent-e35` | QuantAgent-e35 | closed | 2e0f930f 2026-10-03 | merged | - |
| `lote/engine-y-cierre-m1` | - | sin ticket | 1ce70a80 2026-10-08 | no mergeada | 1 |
| `main` | - | sin ticket | 3a15fc22 2026-10-08 | merged | - |

Nota: la clasificación sale solo del comando git. Una rama integrada con squash o rebase aparece como `no mergeada` aunque su trabajo esté en main; no se intentó adivinar.

## 2. Worktrees de la VM

| Path | Rama | Flags | Existe en disco |
|---|---|---|---|
| `/home/azureuser/repos/projects/QuantAgent` | `main` | - | sí |
| `/home/azureuser/repos/autodev-worktrees/QuantAgent/QuantAgent-kkj.10/codex-20260531T024440Z` | `feature/QuantAgent-kkj.10-codex-direct-20260531T024440Z` | - | sí |
| `/home/azureuser/repos/projects/QuantAgent/.claude/worktrees/agent-a9f8d10f6bbbd2787` | `loop/QuantAgent-7c6` | locked | sí |
| `/home/azureuser/repos/projects/QuantAgent/.claude/worktrees/bridge-cse_017axFBScAw8n1EiTEWbsZfP` | `feature/QuantAgent-577-macro-indicadores` | locked | sí |
| `/home/azureuser/repos/projects/QuantAgent/.claude/worktrees/bridge-cse_01HJEHprTyiyKB2kZLrB18rd` | `lote/metricas-auditadas` | - | sí |
| `/home/azureuser/repos/projects/QuantAgent/.claude/worktrees/bridge-cse_01RSviUzNwcWVviS33DehEvV` | `worktree-bridge-cse_01RSviUzNwcWVviS33DehEvV` | locked | sí |
| `/home/azureuser/repos/projects/QuantAgent/.claude/worktrees/bridge-cse_01T9Ld5A9zHUSWbKagKeiSdP` | `worktree-bridge-cse_01T9Ld5A9zHUSWbKagKeiSdP` | locked | sí |
| `/home/azureuser/repos/projects/QuantAgent/.worktrees/feature__QuantAgent-82t-reintegration-20260507T0156Z` | `feature/QuantAgent-82t-reintegration-20260507T0156Z` | - | sí |
| `/home/azureuser/repos/projects/QuantAgent/.worktrees/feature__QuantAgent-zw2-hotfix-ci-failure-in-ci-commit-7cd6670` | `feature/QuantAgent-zw2-hotfix-ci-failure-in-ci-commit-7cd6670` | - | sí |
| `/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.11/direct-20260528T023902Z` | `feature/QuantAgent-kkj.11-configurar-routing-multi-provider-por-ro` | prunable | no |
| `/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.11/implementer-20260528T073800Z` | `feature/QuantAgent-kkj.11-routing-fresh-main` | prunable | no |
| `/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.11/integration-refresh-20260528T1738Z` | `feature/QuantAgent-kkj.11-routing-refresh-20260528` | prunable | no |
| `/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.3/implementer` | `feature/QuantAgent-kkj.3-ux-redise-ar-dashboard-para-ser-environm` | prunable | no |
| `/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.4/implementer` | `feature/QuantAgent-kkj.4-ux-separar-pesta-a-configuration-en-llm` | prunable | no |
| `/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.5/implementer` | `feature/QuantAgent-kkj.5-ux-agregar-help-contextual-en-configurat` | prunable | no |
| `/tmp/claude-1000/.../scratchpad/wt-agent` | `chore/loop-agent-selector` | - | sí |
| `/tmp/claude-1000/.../scratchpad/wt-default` | `chore/loop-default-agy` | - | sí |
| `/tmp/claude-1000/.../scratchpad/wt-docs` | `docs/loop-agy-permissions` | - | sí |
| `/tmp/claude-1000/.../scratchpad/wt-slip` | `chore/plan-slippage-tickets` | - | sí |
| `/tmp/verif-main-89e` | (detached HEAD) | - | sí |

## 3. Clasificación

Las PRs abiertas NO se consultaron (sin API de GitHub); se protegen las ramas citadas en el encargo (`lote/engine-y-cierre-m1`, `loop/QuantAgent-832.1`) y las ramas con worktree.

Total 82 = 51 + 13 + 18 = 82.

### Mergeada, se puede borrar (51)

- `chore/ci-disable-qa-validator`
- `chore/loop-fase0`
- `chore/loop-hermes-entry`
- `feat/Core-State-Mgmnt`
- `feat/LangGraph-Improve`
- `feat/LangGraph-Parallel`
- `feat/Project-Setup`
- `feat/Streamlit-App`
- `feature/QuantAgent-046-hotfix-ci-failure-in-ci-commit-d46effc`
- `feature/QuantAgent-0b5-integrate-positionmonitor-into-tradingsc`
- `feature/QuantAgent-0tk-ci-alembic-migrations`
- `feature/QuantAgent-19f-manual-viewer-salvage`
- `feature/QuantAgent-1p7-save-stategraph-images-to-disk`
- `feature/QuantAgent-1p7-save-stategraph-images-to-disk-and-refer`
- `feature/QuantAgent-339-qa-validator-runtime-real`
- `feature/QuantAgent-375-scope-replay-signal-lookup-to-selected-s`
- `feature/QuantAgent-3o4-implement-tradingscheduler-for-automatic`
- `feature/QuantAgent-3o8-replay-refresh-20260511T1328Z`
- `feature/QuantAgent-40j-fix-missing-benchmark-fixture`
- `feature/QuantAgent-4fm-externalize-config-fresh`
- `feature/QuantAgent-4w4-lookback-windows`
- `feature/QuantAgent-69d-token-time-metrics-refresh`
- `feature/QuantAgent-6t4-structured-output-agents`
- `feature/QuantAgent-7bn`
- `feature/QuantAgent-88h-create-seed-data-script-for-dev-and-qa-d`
- `feature/QuantAgent-8yr-fix-pytest-collection-blockers`
- `feature/QuantAgent-9t5-fix-stale-worktree-path-assumptions-in-wa`
- `feature/QuantAgent-ait-static-util-pandas-freq`
- `feature/QuantAgent-aki-ejecutar-piloto-controlado-de-paper-trad`
- `feature/QuantAgent-bug-stop-tracking-repo-local-venv`
- `feature/QuantAgent-c69-m1-llm-agent-strategy-impl`
- `feature/QuantAgent-c69-m1-llm-agent-strategy-planning`
- `feature/QuantAgent-fg0-test-revision`
- `feature/QuantAgent-g3c-position-reversal`
- `feature/QuantAgent-g3c-position-reversal-fix`
- `feature/QuantAgent-h7d-fix-message`
- `feature/QuantAgent-ia2-complete-universe-management-in-configur`
- `feature/QuantAgent-kkj.1-fix-backtest-run-dup-20260525T173935Z`
- `feature/QuantAgent-kkj.2-agregar-controles-de-scheduler-paper-tra`
- `feature/QuantAgent-kkj.8-crear-strategy-registry-y-parametrizar-s`
- `feature/QuantAgent-kkj.9-agregar-selector-de-estrategia-en-ui-bac`
- `feature/QuantAgent-l8r-fix-trade-pnl-ci-regression`
- `feature/QuantAgent-nrt-fix-backtest-position-monitor-gate-failures`
- `feature/QuantAgent-o2b-fix-azure-provider-gate-failures`
- `feature/QuantAgent-ou3-spx-spy-mapping`
- `feature/QuantAgent-s62-extender-observabilidad-operativa-m-nima`
- `feature/QuantAgent-sft-paper-runtime-hardening-refresh-20260513T175506Z`
- `feature/QuantAgent-um8-implementar-batch-processing-para-llamad`
- `feature/QuantAgent-uzq-fix-tradingscheduler-heartbeat-and-sched`
- `feature/QuantAgent-vje-scheduler-status-and-controls-in-streaml`
- `feature/quantagent-m1-strategy-planning-20260505`

### No tocar (13)

- `feature/QuantAgent-577-macro-indicadores`
- `feature/QuantAgent-kkj.11-configurar-routing-multi-provider-por-ro`
- `feature/QuantAgent-kkj.11-routing-fresh-main`
- `feature/QuantAgent-kkj.11-routing-refresh-20260528`
- `feature/QuantAgent-kkj.3-ux-redise-ar-dashboard-para-ser-environm`
- `feature/QuantAgent-kkj.4-ux-separar-pesta-a-configuration-en-llm`
- `feature/QuantAgent-kkj.5-ux-agregar-help-contextual-en-configurat`
- `feature/plan30`
- `loop/QuantAgent-832.1`
- `loop/QuantAgent-bv8`
- `loop/QuantAgent-e35`
- `lote/engine-y-cierre-m1`
- `main`

### No mergeada, decide Fede (18)

- `feature/QuantAgent-3hs-fix-tradinggraph-reload-mock` — 1 commits — test: stabilize TradingGraph mock after module reload (QuantAgent-3hs)
- `feature/QuantAgent-3o8-implement-replay-execution-mode-reuse-an` — 2 commits — [QuantAgent-3o8] Implement Replay execution mode
- `feature/QuantAgent-3o8-replay-fresh-20260511T023846Z` — 1 commits — [QuantAgent-3o8] Implement replay execution mode with scoped provenance
- `feature/QuantAgent-4fm-externalize-hardcoded-trading-configurat` — 1 commits — [QuantAgent-4fm] Add planning/design docs
- `feature/QuantAgent-69d-implementar-tracking-de-tokens-y-tiempo` — 1 commits — [QuantAgent-69d] Add planning/design docs
- `feature/QuantAgent-6t4-use-with-structured-output-in-pattern-ag` — 1 commits — [QuantAgent-6t4] Add planning/design docs
- `feature/QuantAgent-82t-ci-tests-clean` — 3 commits — docs(QuantAgent-82t): fix tech lead artifact run id
- `feature/QuantAgent-82t-re-enable-unit-tests-in-ci-pipeline-main` — 1 commits — [QuantAgent-82t] Add planning/design docs
- `feature/QuantAgent-e4k-refactor-backtest-to-depend-only-on-orde` — 3 commits — chore: apply ruff --fix quality gate across codebase (QuantAgent-e4k)
- `feature/QuantAgent-hdx-hotfix-ci-failure-in-unit-tests-commit-6` — 1 commits — [QuantAgent-hdx] Implement change
- `feature/QuantAgent-les-support-commissions-in-p-l-calculation-clean` — 6 commits — docs(QuantAgent-les): add tech lead verification evidence
- `feature/QuantAgent-ng1-remove-redundant-main-ci-yml-workflow` — 2 commits — ci(QuantAgent-ng1): remove redundant main CI workflow
- `feature/QuantAgent-sft-paper-runtime-hardening` — 1 commits — docs(planner): add QuantAgent-sft paper runtime hardening plan
- `feature/QuantAgent-vna-m1-strategy-1-triple-screen-strategy-ale` — 1 commits — feat(QuantAgent-vna): implement TripleScreenStrategy with tests and backtest fix
- `feature/QuantAgent-vna-triple-screen-strategy-alexander-elder` — 1 commits — [QuantAgent-vna] Add planning/design docs
- `feature/QuantAgent-x8u-hotfix-ci-failure-in-ci-commit-f8cf1e5` — 2 commits — [QuantAgent-x8u] Implement change
- `gh-pages` — 21 commits — deploy: 1429e3364636ad1b7f24ee501e6e7ed2333f1a43
- `integration/QuantAgent-82t-20260508T213950Z` — 3 commits — docs(QuantAgent-82t): record blocked integration review

## 4. Comandos (NO ejecutados)

Nadie ejecutó estos comandos; solo los decide y corre el dueño.

```bash
git push origin --delete chore/ci-disable-qa-validator
git push origin --delete chore/loop-fase0
git push origin --delete chore/loop-hermes-entry
git push origin --delete feat/Core-State-Mgmnt
git push origin --delete feat/LangGraph-Improve
git push origin --delete feat/LangGraph-Parallel
git push origin --delete feat/Project-Setup
git push origin --delete feat/Streamlit-App
git push origin --delete feature/QuantAgent-046-hotfix-ci-failure-in-ci-commit-d46effc
git push origin --delete feature/QuantAgent-0b5-integrate-positionmonitor-into-tradingsc
git push origin --delete feature/QuantAgent-0tk-ci-alembic-migrations
git push origin --delete feature/QuantAgent-19f-manual-viewer-salvage
git push origin --delete feature/QuantAgent-1p7-save-stategraph-images-to-disk
git push origin --delete feature/QuantAgent-1p7-save-stategraph-images-to-disk-and-refer
git push origin --delete feature/QuantAgent-339-qa-validator-runtime-real
git push origin --delete feature/QuantAgent-375-scope-replay-signal-lookup-to-selected-s
git push origin --delete feature/QuantAgent-3o4-implement-tradingscheduler-for-automatic
git push origin --delete feature/QuantAgent-3o8-replay-refresh-20260511T1328Z
git push origin --delete feature/QuantAgent-40j-fix-missing-benchmark-fixture
git push origin --delete feature/QuantAgent-4fm-externalize-config-fresh
git push origin --delete feature/QuantAgent-4w4-lookback-windows
git push origin --delete feature/QuantAgent-69d-token-time-metrics-refresh
git push origin --delete feature/QuantAgent-6t4-structured-output-agents
git push origin --delete feature/QuantAgent-7bn
git push origin --delete feature/QuantAgent-88h-create-seed-data-script-for-dev-and-qa-d
git push origin --delete feature/QuantAgent-8yr-fix-pytest-collection-blockers
git push origin --delete feature/QuantAgent-9t5-fix-stale-worktree-path-assumptions-in-wa
git push origin --delete feature/QuantAgent-ait-static-util-pandas-freq
git push origin --delete feature/QuantAgent-aki-ejecutar-piloto-controlado-de-paper-trad
git push origin --delete feature/QuantAgent-bug-stop-tracking-repo-local-venv
git push origin --delete feature/QuantAgent-c69-m1-llm-agent-strategy-impl
git push origin --delete feature/QuantAgent-c69-m1-llm-agent-strategy-planning
git push origin --delete feature/QuantAgent-fg0-test-revision
git push origin --delete feature/QuantAgent-g3c-position-reversal
git push origin --delete feature/QuantAgent-g3c-position-reversal-fix
git push origin --delete feature/QuantAgent-h7d-fix-message
git push origin --delete feature/QuantAgent-ia2-complete-universe-management-in-configur
git push origin --delete feature/QuantAgent-kkj.1-fix-backtest-run-dup-20260525T173935Z
git push origin --delete feature/QuantAgent-kkj.2-agregar-controles-de-scheduler-paper-tra
git push origin --delete feature/QuantAgent-kkj.8-crear-strategy-registry-y-parametrizar-s
git push origin --delete feature/QuantAgent-kkj.9-agregar-selector-de-estrategia-en-ui-bac
git push origin --delete feature/QuantAgent-l8r-fix-trade-pnl-ci-regression
git push origin --delete feature/QuantAgent-nrt-fix-backtest-position-monitor-gate-failures
git push origin --delete feature/QuantAgent-o2b-fix-azure-provider-gate-failures
git push origin --delete feature/QuantAgent-ou3-spx-spy-mapping
git push origin --delete feature/QuantAgent-s62-extender-observabilidad-operativa-m-nima
git push origin --delete feature/QuantAgent-sft-paper-runtime-hardening-refresh-20260513T175506Z
git push origin --delete feature/QuantAgent-um8-implementar-batch-processing-para-llamad
git push origin --delete feature/QuantAgent-uzq-fix-tradingscheduler-heartbeat-and-sched
git push origin --delete feature/QuantAgent-vje-scheduler-status-and-controls-in-streaml
git push origin --delete feature/quantagent-m1-strategy-planning-20260505
git worktree prune
```
