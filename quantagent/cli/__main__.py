"""Entry point for QuantAgent CLI."""

import click

from .backtest import backtest_group
from .data import data_group
from .profile import profile_group


@click.group(help="QuantAgent command-line interface.")
def cli() -> None:
    """Top-level CLI group."""


# Register subcommands
cli.add_command(profile_group, name="profile")
cli.add_command(backtest_group, name="backtest")
cli.add_command(data_group, name="data")


if __name__ == "__main__":
    cli()
