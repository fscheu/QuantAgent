import re
from itertools import takewhile
from click.core import Group
from pathlib import Path

from quantagent.cli.__main__ import cli


def test_funnel_command_paths_exist_in_cli():
    commands = re.findall(r"`python -m quantagent\.cli ([^`]+)`", Path("docs/manual/funnel.md").read_text())
    assert commands
    for invocation in commands:
        command = cli
        route = list(takewhile(lambda token: not token.startswith(("<", "--")), invocation.split()))
        for name in route:
            assert isinstance(command, Group), f"unexpected route segment {name!r} in {invocation!r}"
            assert name in command.commands, f"unknown command {name!r} in {invocation!r}"
            command = command.commands[name]
