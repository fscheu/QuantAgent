from pathlib import Path

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "main-ci-deploy.yml"


def read_workflow() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_healthcheck_and_validator_target_use_streamlit_runtime_8501():
    workflow = read_workflow()

    assert "curl -fsS http://127.0.0.1:8501/_stcore/health" in workflow
    assert "/home/azureuser/repos/agents/qa-validator-poc/configs/local-streamlit.yaml" in workflow
    assert "grep -q '8501' \"$CONFIG\"" in workflow
    assert "Deployed QA target: http://127.0.0.1:8501" in workflow


def test_validator_result_step_distinguishes_success_partial_and_artifacts():
    workflow = read_workflow()

    assert "qa_verified = str(status == 'SUCCESS').lower()" in workflow
    assert "qa_partial = str(status == 'PARTIAL').lower()" in workflow
    assert "f.write(f'qa_validator_result_status={status}\\n')" in workflow
    assert "f.write(f'qa_verified={qa_verified}\\n')" in workflow
    assert "f.write(f'qa_partial={qa_partial}\\n')" in workflow
    assert "f.write(f'validator_artifacts_count={artifacts_count}\\n')" in workflow
    assert "f.write(f'validator_artifacts_list={artifacts_list}\\n')" in workflow
    assert "if p.is_file() and p.suffix in ('.png', '.jpg', '.html', '.log'):" in workflow


def test_workflow_uploads_validator_artifacts_and_reports_metadata_to_hermes():
    workflow = read_workflow()

    assert "name: Upload QA validator artifacts" in workflow
    assert "uses: actions/upload-artifact@v4" in workflow
    assert "name: qa-validator-${{ github.run_id }}-${{ steps.qa_validator_result.outputs.qa_validator_result_status || 'MISSING' }}" in workflow
    assert "path: /home/azureuser/repos/agents/qa-validator-poc/runs/poc-qa-validator-local-streamlit-8501/" in workflow
    assert "retention-days: 30" in workflow
    assert '\"qa_target_alignment_outcome\": os.environ.get(\"QA_TARGET_ALIGNMENT_OUTCOME\", \"\")' in workflow
    assert '\"qa_partial\": os.environ.get(\"QA_PARTIAL\", \"false\")' in workflow
    assert '\"validator_artifacts_count\": os.environ.get(\"VALIDATOR_ARTIFACTS_COUNT\", \"0\")' in workflow
    assert '\"validator_artifacts_list\": os.environ.get(\"VALIDATOR_ARTIFACTS_LIST\", \"\")' in workflow
