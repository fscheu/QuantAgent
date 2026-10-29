"""Entry point for QuantAgent CLI."""

import click

from .backtest import backtest_group
from .data import data_group
from .intentos import intentos_group
from .profile import profile_group
from .reconcile import trading_group


@click.group(help="QuantAgent command-line interface.", epilog="Run 'quantagent COMMAND --help' for command details.")
def cli() -> None:
    """Top-level CLI group."""


# Register subcommands
cli.add_command(profile_group, name="profile")
cli.add_command(backtest_group, name="backtest")
cli.add_command(data_group, name="data")
cli.add_command(trading_group, name="trading")
cli.add_command(intentos_group, name="intentos")


if __name__ == "__main__":
    cli()
