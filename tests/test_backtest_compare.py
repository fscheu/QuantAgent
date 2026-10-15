import pytest
from click.testing import CliRunner

from quantagent.cli.backtest import backtest_group


@pytest.mark.parametrize("command", ["run", "compare"])
def test_cli_llenado_invalido(command):
    result = CliRunner().invoke(backtest_group, [command, "--llenado", "otro"])
    assert result.exit_code == 2
    assert "Invalid value for '--llenado'" in result.output
