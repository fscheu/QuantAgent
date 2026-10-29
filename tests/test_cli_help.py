import click

from quantagent.cli.__main__ import cli


def test_cli_commands_groups_and_options_have_english_help():
    forbidden = set("áéíóúñ¿¡")

    def check(command):
        assert command.help and not forbidden.intersection(command.help.lower()), command.name
        assert not forbidden.intersection((command.epilog or "").lower()), command.name
        for option in command.params:
            if isinstance(option, click.Option):
                assert option.help and not forbidden.intersection(option.help.lower()), (
                    command.name, option.name
                )
        if isinstance(command, click.Group):
            for child in command.commands.values():
                check(child)

    check(cli)
