"""Render the Click command tree as Markdown pages without writing them."""

from __future__ import annotations

import click

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
        if isinstance(command, click.Group):
            for child_name, child in sorted(command.commands.items()):
                parts.append(_render_command(child, child_name, child_context, depth + 1))
        return "\n".join(parts)


def render_cli_reference(root: click.Group = cli) -> dict[str, str]:
    """Return deterministic Markdown for each top-level command, in memory."""
    pages = {}
    with root.make_context("quantagent", [], resilient_parsing=True) as context:
        pages["index.md"] = "# QuantAgent CLI reference\n\n```text\n" + root.get_help(context).rstrip() + "\n```\n"
        for name, command in sorted(root.commands.items()):
            pages[f"{name}.md"] = f"# `{name}` command reference\n\n" + _render_command(command, name, context)
    return pages


if __name__ == "__main__":
    for path, markdown in render_cli_reference().items():
        print(f"--- {path} ---")
        print(markdown)
