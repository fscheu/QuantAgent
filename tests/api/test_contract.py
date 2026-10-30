import json
from pathlib import Path

from click.testing import CliRunner

from apps.api.app import app
from apps.api.cli import api_group


CONTRACT = Path(__file__).resolve().parents[2] / "docs/api/openapi.json"


def serialize_schema(schema):
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def test_live_openapi_matches_frozen_contract():
    assert serialize_schema(app.openapi()) == CONTRACT.read_text(), (
        "OpenAPI contract differs; regenerá con: python -m quantagent.cli api schema"
    )


def test_schema_check_accepts_match_and_rejects_difference(tmp_path):
    runner = CliRunner()
    output = tmp_path / "openapi.json"
    output.write_text(serialize_schema(app.openapi()))

    assert runner.invoke(api_group, ["schema", "--check", "--output", str(output)]).exit_code == 0

    output.write_text(output.read_text().replace('"title": "QuantAgent API"', '"title": "Altered API"'))
    result = runner.invoke(api_group, ["schema", "--check", "--output", str(output)])
    assert result.exit_code == 1
