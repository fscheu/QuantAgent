# Deploy Verification — GitHub Actions Webhook

**Timestamp:** 2026-05-07T13:22:30Z  
**Webhook source:** GitHub Actions `deploy_finished`  
**Tech Lead:** Hermes  
**Verification mode:** Post-deploy webhook (failure)

---

## Webhook Identity

| Key | Value |
|-----|-------|
| **Repo** | `fscheu/QuantAgent` |
| **Workflow** | Main CI + Deploy QA |
| **Job** | `deploy-qa` |
| **Environment** | qa |
| **Branch** | main |
| **Ref** | `refs/heads/main` |
| **SHA** | `e273f215e9fb4d57497828ee23fe8790d663210d` |
| **Actor** | fscheu |
| **Conclusion** | **failure** |
| **Run ID** | [25496309598](https://github.com/fscheu/QuantAgent/actions/runs/25496309598) |
| **Run attempt** | 1 |
| **Deployed URL** | https://qa.fedes.dev |
| **Deploy step** | failure |
| **Health check step** | skipped |

---

## GitHub Source of Truth Verification

```bash
gh run view 25496309598 --json conclusion,status,headBranch,headSha,jobs
```

**GitHub conclusion:** `failure` ✓ (matches webhook)  
**GitHub status:** `completed`  
**Head branch:** `main` ✓  
**Head SHA:** `e273f215e9fb4d57497828ee23fe8790d663210d` ✓

**Jobs:**
1. **CI (Lint + Tests)** → `success` ✅
2. **Deploy to QA** → `failure` ❌

---

## Commit Context

**SHA e273f215:**
```
chore: sync beads state after QuantAgent-3hs close

 .beads/issues.jsonl | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

**Classification:** Sync commit (BEADS bookkeeping only, no functional code).

**Preceding merge commit (e7723feb):**
```
[QuantAgent-3hs] Fix checkpointing runtime gate failures

Ticket: QuantAgent-3hs
Title: Fix checkpointing runtime gate failures after Azure/backtest blockers
```

**Real verification target:** `e7723feb` (functional merge), not `e273f215` (bookkeeping).

---

## Failure Analysis

### Step-level breakdown

| Job | Step | Outcome | Duration |
|-----|------|---------|----------|
| CI (Lint + Tests) | All steps | success | ~3 min |
| Deploy to QA | Checkout | success | 1s |
| Deploy to QA | **Deploy to QA locally** | **failure** | ~8 min |
| Deploy to QA | Health check | skipped | — |
| Deploy to QA | Notify Hermes Tech Lead | success | 1s |

### Root cause (log tail)

```
Step 5/8 : COPY . .
write /app/.worktrees/feature__QuantAgent-94d-add-backtest-run-id-isolation-to-activep/.venv/lib/python3.12/site-packages/scipy/stats/__pycache__/_stats_py.cpython-311.pyc: no space left on device
Service 'api' failed to build : Build failed
##[error]Process completed with exit code 1.
```

**Failure signature:** `no space left on device` during Docker `COPY . .` step inside a worktree `.venv/`.

---

## Classification

| Dimension | Value |
|-----------|-------|
| **Category** | Infraestructura / Docker capacity |
| **Subcategory** | Disk exhaustion during build |
| **Severity** | Non-blocking (transient) |
| **Scope** | Self-hosted runner host filesystem |
| **Code impact** | None — functional diff is sound |

---

## Infrastructure State Snapshot

### Runner disk usage

```bash
df -h /
Filesystem      Size  Used Avail Use% Mounted on
/dev/root        61G   58G  3.7G  94% /
```

**Root disk:** 94% full, 3.7 GB available.

### Docker storage

```bash
docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          15        1         28.01GB   24.67GB (88%)
Containers      1         1         63B       0B (0%)
Local Volumes   4         1         210.6MB   135.1MB (64%)
Build Cache     0         0         0B        0B
```

**Reclaimable space:** 24.67 GB from dangling images (88% of total image storage).

### Build context bloat

```bash
du -sh /home/azureuser/repos/projects/QuantAgent/.worktrees/
1.3G    .worktrees/
```

**Active worktrees in main checkout:**
- `.worktrees/feature__QuantAgent-82t-reintegration-20260507T0156Z`
- `.worktrees/feature__QuantAgent-zw2-hotfix-ci-failure-in-ci-commit-7cd6670`

Each contains a full `.venv/` with ~500+ MB of Python libs.

**.dockerignore coverage:**
- `.venv` ✓ ignored
- `.worktrees/` ✗ **NOT ignored** → copied to build context

**Failure mode:** Docker daemon tries to copy 1.3 GB of worktree + venv state into the build context, exhausts remaining 3.7 GB disk headroom during layer write.

---

## Action Plan

### 1. Free reclaimable Docker storage (P0)

```bash
docker image prune -af
docker container prune -f
docker volume prune -f
```

**Expected recovery:** ~24.67 GB.

### 2. Add .worktrees/ to .dockerignore (P0)

**Patch `.dockerignore`:**
```diff
 # Python
 __pycache__
 *.pyc
 *.pyo
 *.pyd
 .Python
 *.so
 *.egg
 *.egg-info
 dist
 build
 .venv
 venv
+.worktrees
 .pytest_cache
 .mypy_cache
 .ruff_cache
```

**Reasoning:**
- Worktrees are local-only dev state, never needed in container images.
- Each worktree carries its own `.venv/`, bloating build context by ~GB per worktree.
- This is a recurring issue pattern on multi-ticket autodev runs.

### 3. Rerun failed workflow (P0)

After cleanup + `.dockerignore` fix:
```bash
gh workflow run 'Main CI + Deploy QA' --ref main
```

Or manual trigger from Actions UI.

**Expected outcome:** Build succeeds with freed disk space + pruned build context.

### 4. Verify QA deploy health (P1)

Once workflow succeeds:
```bash
curl -s https://qa.fedes.dev/health | jq
```

Expect:
```json
{
  "status": "healthy",
  "version": "...",
  "commit": "e273f215..."
}
```

---

## Anomalies (Non-Blocking)

### GitHub Actions Telegram notification defect (known)

**Symptom:** `Notify Telegram on success` step in CI job reported `success`, but no Telegram delivery actually happened.

**Evidence:**
```json
{
  "name": "Notify Telegram on success",
  "conclusion": "success",
  "status": "completed"
}
```

No corresponding message received in Telegram (expected format: "✅ CI passed — fscheu/QuantAgent main@e273f215").

**Classification:** Known non-blocking GitHub Actions notification bug; does not affect deploy logic or health.

**Reference:** `~/.hermes/skills/fede/tech-lead-autodev/references/qa-deploy-notification-anomalies.md`

**Resolution:** This is a GitHub Actions/notification integration quirk; not a code regression. Monitor for pattern escalation; no immediate action required.

---

## Decision

**Do NOT treat this as a code regression.**

This is a **self-hosted runner capacity exhaustion** triggered by:
1. Docker image bloat (24.67 GB reclaimable but not pruned)
2. Build context bloat (1.3 GB `.worktrees/` not excluded from COPY)
3. 94% root disk usage before build started

**Next step:** Execute P0 action plan (prune + .dockerignore + rerun).

**Merge status:** The underlying merge commit `e7723feb` (QuantAgent-3hs) is functionally sound; CI passed, code diff is correct. Failure is infrastructure-side only.

---

## Follow-Up

- [ ] Prune Docker images
- [ ] Add `.worktrees` to `.dockerignore`
- [ ] Commit `.dockerignore` fix: `chore(docker): exclude .worktrees from build context`
- [ ] Rerun workflow 25496309598 or trigger fresh main deploy
- [ ] Verify QA health endpoint after successful deploy
- [ ] Monitor runner disk usage trend; consider scheduled prune cron if pattern recurs

---

**Tech Lead verification completed:** 2026-05-07T13:35Z  
**Artifact path:** `docs/envelopes/deploy-verification-20260507T132230Z.md`
