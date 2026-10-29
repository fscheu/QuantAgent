"""Render the Click command tree as Markdown pages without writing them."""

from __future__ import annotations

import click
from pathlib import Path

from quantagent.cli.__main__ import cli

_UNSET = getattr(click.core, "UNSET", object())


def _render_command(command: click.Command, name: str, context: click.Context, depth: int = 2) -> str:
    """Render one command and its descendants using Click's current help output."""
    with command.make_context(name, [], parent=context, resilient_parsing=True) as child_context:
        heading = "#" * depth
        parts = [f"{heading} `{child_context.command_path}`", "", "```text", command.get_help(child_context).rstrip(), "```", ""]
        options = [param for param in command.params if isinstance(param, click.Option)]
        if options:
            parts.extend(["Options and defaults:", ""])
            for option in options:
                names = ", ".join(option.opts + option.secondary_opts)
                default = option.get_default(child_context)
                if default is _UNSET:
                    default = None
                parts.append(f"- `{names}` — default: `{default!r}`")
            parts.append("")
        return "\n".join(parts)


def render_cli_reference(root: click.Group = cli, output_dir: Path | None = None) -> dict[str, str]:
    """Return one deterministic Markdown page per command group, optionally writing them."""
    pages = {}
    with root.make_context("quantagent", [], resilient_parsing=True) as context:
        pages["index.md"] = "# QuantAgent CLI reference\n\n```text\n" + root.get_help(context).rstrip() + "\n```\n"
        def add_group(group: click.Group, name: str, parent: click.Context, path: tuple[str, ...]) -> None:
            with group.make_context(name, [], parent=parent, resilient_parsing=True) as group_context:
                parts = [f"# `{group_context.command_path}` command reference", "", "```text", group.get_help(group_context).rstrip(), "```", ""]
                for child_name, child in sorted(group.commands.items()):
                    parts.append(_render_command(child, child_name, group_context, 2))
                pages["/".join((*path, f"{name}.md"))] = "\n".join(parts)
                for child_name, child in sorted(group.commands.items()):
                    if isinstance(child, click.Group):
                        add_group(child, child_name, group_context, (*path, name))

        for name, command in sorted(root.commands.items()):
            if isinstance(command, click.Group):
                add_group(command, name, context, ())
    if output_dir is not None:
        for path, markdown in pages.items():
            destination = output_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(markdown)
    return pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the Click command tree as Markdown pages.")
    parser.add_argument("--output-dir", type=Path, help="Write generated pages to this directory.")
    for path, markdown in render_cli_reference(output_dir=parser.parse_args().output_dir).items():
        print(f"--- {path} ---")
        print(markdown)
