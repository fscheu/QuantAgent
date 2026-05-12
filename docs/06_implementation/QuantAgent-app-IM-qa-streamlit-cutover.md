# QuantAgent-app Implementation Notes

**Issue:** QuantAgent-app — QA Streamlit cutover + Cloudflare tunnel alignment
**Mode:** Manual infra/application validation
**Branch:** main

## Summary

The QA environment now serves the Streamlit UI on port 8501 instead of the legacy Flask health server on port 8001.

The deployment contract was updated so `deploy_finished` remains the end-of-job webhook carrying deploy and health metadata for the 8501 runtime, while `qa_verified` is a derived post-deploy validation state inferred from the validator outcome attached to that deploy event.

Cloudflare was cut over so `qa.fedes.dev` now routes to `localhost:8501`, and the temporary `qa-ui.fedes.dev` hostname was removed from the published application routes.

## Change made

### Runtime / container

- `Dockerfile.qa`
  - `EXPOSE` changed from 8001 to 8501.
  - Final command changed from Flask health server to:
    - `streamlit run apps/streamlit/app.py --server.headless=true --server.address=0.0.0.0 --server.port=8501`
  - Docker healthcheck now targets `http://localhost:8501/_stcore/health`.

- `docker-compose.qa.yml`
  - Port mapping changed to `8501:8501`.
  - Service healthcheck changed to `http://localhost:8501/_stcore/health`.

### CI / deploy contract

- `.github/workflows/main-ci-deploy.yml`
  - Added aggressive Docker cleanup before rebuild:
    - `docker image prune -af`
    - `docker container prune -f`
    - `docker builder prune -af`
  - Resets the fixed-path validator run directory before each deploy so stale artifacts cannot be reused across runs.
  - QA health gate changed from the old Flask endpoint to:
    - `http://127.0.0.1:8501/_stcore/health`
  - Health retry window expanded to 18 attempts with 10-second pauses.
  - Added a pinned post-deploy QA validator step using:
    - `/home/azureuser/repos/agents/qa-validator-poc/configs/local-streamlit.yaml`
  - Added integrity checks for the external validator inputs before execution:
    - `runner.py`
    - `configs/local-streamlit.yaml`
    - `prompt_template.md`
    - `docs/local-streamlit-target.md`
  - After the validator runs, the workflow reads `result.json` and derives the verification verdict from `status` instead of trusting only the step exit status.
  - Webhook payload to Hermes now includes:
    - `deploy_step_outcome`
    - `health_step_outcome`
    - `qa_validator_step_outcome`
    - `qa_validator_result_status`
    - `qa_verified`

### Cloudflare / tunnel

- Local cloudflared config was aligned to `service: http://localhost:8501` for `qa.fedes.dev`.
- The effective remote tunnel configuration was later updated from Cloudflare Dashboard.
- Final observed tunnel config in `journalctl -u cloudflared`:
  - `qa.fedes.dev -> http://localhost:8501`
  - `qa-ui.fedes.dev` removed

## Operational contract

### 1. `deploy_finished`

`deploy_finished` is the webhook emitted at the end of the QA deploy job. It always carries the deploy and health outcomes, so the receiver must inspect those fields instead of assuming success from the event name alone.

Interpretation rules:
- `deploy_step_outcome=success` means compose rebuild/up completed.
- `health_step_outcome=success` means the deployed QA service answered on `127.0.0.1:8501/_stcore/health`.
- When both are `success`, the new QA runtime is up locally on the VM.
- `conclusion` remains the GitHub job conclusion.

This is the correct event for reporting: "the deploy job finished, with explicit runtime-health metadata attached."

### 2. `qa_verified`

`qa_verified` is a stricter state derived from `deploy_finished`, not a separate webhook event.

It means the browser-oriented validator ran against the local QA target and produced a successful verdict with evidence.

Interpretation rules:
- `event_type` remains `deploy_finished`
- `qa_validator_result_status=SUCCESS` => treat the deploy as `qa_verified`
- `qa_validator_result_status=PARTIAL|BLOCKED|FAIL|INVALID|MISSING` => do not treat the deploy as `qa_verified`
- `qa_validator_step_outcome` remains useful operational metadata, but it is not the canonical functional verdict.
- The validator remains non-blocking for deploy, by design.

This is the correct signal for: "the deployed QA UI was functionally verified beyond container liveness."

### 3. Why validation stays local

`qa.fedes.dev` is protected by Cloudflare Access, so unattended browser automation should validate against `127.0.0.1:8501`, not against the public hostname.

Public hostname purpose:
- human access through Cloudflare Access
- externally shareable QA entrypoint

Validator purpose:
- deterministic post-deploy functional verification without Access auth coupling

## Verification evidence

### Local runtime

Verified on the VM:
- `curl http://127.0.0.1:8501/_stcore/health` -> `200 OK`, body `ok`
- `curl http://127.0.0.1:8001/health` -> connection refused

This confirms the old 8001 path is no longer serving QA and the active runtime is 8501.

### Validator

Validated successfully with:
- `python3 /home/azureuser/repos/agents/qa-validator-poc/runner.py --config /home/azureuser/repos/agents/qa-validator-poc/configs/local-streamlit.yaml`

Evidence directory:
- `/home/azureuser/repos/agents/qa-validator-poc/runs/poc-qa-validator-local-streamlit-8501/`

Observed result:
- `SUCCESS`
- no browser console errors
- screenshot artifact generated by the validator

### Public hostname / Access

Verified externally from the VM:
- `curl -I https://qa.fedes.dev` -> `HTTP/2 302`
- redirect target: Cloudflare Access login for `qa.fedes.dev`

Verified with browser navigation:
- URL resolved to Cloudflare Access login
- page title: `Sign in ・ Cloudflare Access`

This is expected and confirms the public hostname is live behind Access.

### Tunnel routing evidence

Observed in `journalctl -u cloudflared` after the dashboard change:
- configuration version 3 updated `qa.fedes.dev` to `http://localhost:8501`
- configuration version 4 removed `qa-ui.fedes.dev`

That is the strongest direct evidence that the active tunnel route now targets port 8501.

## Known limitations / notes

- Cloudflare Access prevents using `qa.fedes.dev` as the primary automated validation target without an authenticated access flow.
- The Cloudflare Dashboard public-hostname configuration can change the effective tunnel routing independently of the local file, so tunnel logs should be checked when diagnosing route drift.
- Disk pressure on the VM is still a real constraint; the prune steps are intentional, not cosmetic.

## User manual impact

Internal QA deploy/runtime contract only. No end-user product behavior changed beyond QA now exposing the Streamlit UI instead of the old health-only service.
