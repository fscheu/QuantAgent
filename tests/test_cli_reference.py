from pathlib import Path

from click.core import Group, Option

from quantagent.cli.__main__ import cli
from scripts.gen_cli_reference import render_cli_reference


def test_cli_reference_is_stable_complete_and_matches_approved_pages():
    pages = render_cli_reference()
    assert pages == render_cli_reference()

    def check_help(command, name, context):
        with command.make_context(name, [], parent=context, resilient_parsing=True) as child:
            help_text = command.get_help(child)
            assert "--help" in help_text
            assert "Show this message and exit." in help_text
            for param in command.params:
                if isinstance(param, Option):
                    assert param.opts[0] in help_text
            if isinstance(command, Group):
                for child_name, child_command in sorted(command.commands.items()):
                    check_help(child_command, child_name, child)

    with cli.make_context("quantagent", [], resilient_parsing=True) as context:
        check_help(cli, "quantagent", context)

    approved = Path("docs/manual/reference/cli")
    for filename, rendered in pages.items():
        reference = approved / filename
        if reference.exists():
            assert reference.read_text() == rendered
