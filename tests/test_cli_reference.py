from pathlib import Path
import re

import click
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

    approved = Path(__file__).resolve().parents[1] / "docs/manual/reference/cli"
    assert approved.joinpath("index.md").is_file()
    assert {f"{name}.md" for name in cli.commands} <= pages.keys()
    readme = (approved / "README.md").read_text()
    links = set(re.findall(r"\]\(([^)]+)\)", readme))
    assert set(pages) == links
    assert all((approved / link).is_file() for link in links)
    for filename, rendered in pages.items():
        reference = approved / filename
        if reference.exists():
            assert reference.read_text() == rendered


def test_cli_reference_writes_a_page_for_each_group(tmp_path):
    @click.group()
    def root():
        pass

    @root.group()
    def nested():
        pass

    @nested.group()
    def run():
        pass

    @run.command()
    @click.option("--count", default=3, help="Number of items.")
    def execute(count):
        pass

    pages = render_cli_reference(root, tmp_path)

    assert "nested.md" in pages
    assert "nested/run.md" in pages
    assert all((tmp_path / path).read_text() == page for path, page in pages.items())
    assert "--help" in pages["nested.md"]
    assert "--count" in pages["nested/run.md"]
    assert "default: `3`" in pages["nested/run.md"]
