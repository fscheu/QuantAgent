"""Entry point for QuantAgent CLI."""

import click

from .profile import profile_group


@click.group(help="QuantAgent command-line interface.")
def cli() -> None:
    """Top-level CLI group."""


# Register subcommands
cli.add_command(profile_group, name="profile")


if __name__ == "__main__":
    cli()
