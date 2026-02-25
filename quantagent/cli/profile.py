"""CLI commands for managing StrategyConfig profiles."""

from __future__ import annotations

from typing import Optional

import click
from sqlalchemy.exc import IntegrityError

from quantagent.models import StrategyConfig

from . import utils

try:  # pragma: no cover - dependency is declared but guard for safety
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("tabulate is required for the QuantAgent CLI") from exc


def _validate_identifier(name: Optional[str], profile_id: Optional[int]) -> None:
    if not name and profile_id is None:
        raise click.UsageError("Provide a profile NAME argument or --id.")
    if name and profile_id is not None:
        raise click.UsageError("Use either NAME or --id, not both.")


def _fetch_profile(name: Optional[str], profile_id: Optional[int]) -> StrategyConfig:
    """Load a profile by name or id, raising a ClickException when missing."""

    with utils.session_scope() as session:
        query = session.query(StrategyConfig)
        if profile_id is not None:
            profile = query.filter(StrategyConfig.id == profile_id).one_or_none()
        else:
            profile = query.filter(StrategyConfig.name == name).one_or_none()

        if profile is None:
            identifier = name if name else profile_id
            raise click.ClickException(f"Profile not found: {identifier}")

        session.expunge(profile)
        return profile


@click.group(help="Manage StrategyConfig profiles (list, show, create, update, delete).")
def profile_group() -> None:
    """Root command group for profile operations."""


@profile_group.command("list", help="List profiles with optional filtering.")
@click.option("--kind", type=click.Choice(["portfolio", "risk", "combined"]), help="Filter by profile kind.")
@click.option("--name-like", "name_like", help="Filter by SQL LIKE pattern (e.g., %portfolio%).")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON instead of a table.")
def list_profiles(kind: Optional[str], name_like: Optional[str], as_json: bool) -> None:
    with utils.session_scope() as session:
        query = session.query(StrategyConfig)
        if kind:
            query = query.filter(StrategyConfig.kind == kind)
        if name_like:
            pattern = name_like if "%" in name_like else f"%{name_like}%"
            query = query.filter(StrategyConfig.name.like(pattern))
        profiles = query.order_by(StrategyConfig.created_at.desc()).all()

    if as_json:
        utils.echo_json([utils.serialize_profile(p) for p in profiles])
        return

    if not profiles:
        click.echo("No profiles found.")
        return

    rows = [
        [
            profile.id,
            profile.name,
            profile.kind,
            profile.version,
            utils.format_timestamp(profile.created_at),
            utils.format_timestamp(profile.updated_at),
        ]
        for profile in profiles
    ]
    headers = ["id", "name", "kind", "version", "created_at", "updated_at"]
    click.echo(tabulate(rows, headers=headers, tablefmt="github"))


@profile_group.command("show", help="Show details for a profile by name or id.")
@click.argument("name", required=False)
@click.option("--id", "profile_id", type=int, help="Lookup profile by numeric id.")
@click.option("--json", "as_json", is_flag=True, help="Return data as JSON.")
def show_profile(name: Optional[str], profile_id: Optional[int], as_json: bool) -> None:
    _validate_identifier(name, profile_id)
    profile = _fetch_profile(name, profile_id)

    if as_json:
        utils.echo_json(utils.serialize_profile(profile))
        return

    click.echo(f"ID: {profile.id}")
    click.echo(f"Name: {profile.name}")
    click.echo(f"Kind: {profile.kind}")
    click.echo(f"Version: {profile.version}")
    click.echo(f"Created: {utils.format_timestamp(profile.created_at)}")
    click.echo(f"Updated: {utils.format_timestamp(profile.updated_at)}")
    click.echo("json_config:")
    utils.echo_json(profile.json_config)


@profile_group.command("create", help="Create a new profile from JSON input.")
@click.option("--config", "config_text", help="Inline JSON payload. If omitted, read from stdin.")
def create_profile(config_text: Optional[str]) -> None:
    payload = utils.load_json_payload(config_text)
    utils.ensure_required_fields(payload, ["name", "kind", "json_config"])

    json_config = utils.ensure_json_object(payload["json_config"], "json_config")
    version = payload.get("version", 1)

    profile = StrategyConfig(
        name=payload["name"],
        kind=payload["kind"],
        json_config=json_config,
        version=version,
    )

    try:
        with utils.session_scope() as session:
            session.add(profile)
            session.flush()
            session.refresh(profile)
    except IntegrityError as exc:
        raise click.ClickException(
            f"Profile with name '{payload['name']}' already exists. Use 'update' or choose a different name."
        ) from exc

    click.echo(f"Profile '{profile.name}' created (id={profile.id}, version={profile.version}).")


@profile_group.command("update", help="Update fields on an existing profile.")
@click.argument("name", required=False)
@click.option("--id", "profile_id", type=int, help="Lookup profile by id instead of name.")
@click.option("--config", "config_text", help="JSON payload with fields to update.")
@click.option("--replace", is_flag=True, help="Replace json_config instead of merging.")
@click.option("--keep-version", is_flag=True, help="Do not increment the version number.")
def update_profile(
    name: Optional[str],
    profile_id: Optional[int],
    config_text: Optional[str],
    replace: bool,
    keep_version: bool,
) -> None:
    _validate_identifier(name, profile_id)
    payload = utils.load_json_payload(config_text)
    if not payload:
        raise click.ClickException("Payload is empty; specify at least one field to update.")

    with utils.session_scope() as session:
        query = session.query(StrategyConfig)
        if profile_id is not None:
            profile = query.filter(StrategyConfig.id == profile_id).one_or_none()
        else:
            profile = query.filter(StrategyConfig.name == name).one_or_none()

        if profile is None:
            identifier = name if name else profile_id
            raise click.ClickException(f"Profile not found: {identifier}")

        if "name" in payload:
            profile.name = payload["name"]
        if "kind" in payload:
            profile.kind = payload["kind"]
        if "version" in payload:
            profile.version = payload["version"]

        if "json_config" in payload:
            json_config_update = utils.ensure_json_object(payload["json_config"], "json_config")
            if replace:
                profile.json_config = json_config_update
            else:
                current = dict(profile.json_config or {})
                current.update(json_config_update)
                profile.json_config = current

        if not keep_version and "version" not in payload:
            profile.version = (profile.version or 0) + 1

        session.add(profile)
        session.flush()
        session.refresh(profile)

    click.echo(f"Profile '{profile.name}' updated (version={profile.version}).")


@profile_group.command("delete", help="Delete a profile by name or id.")
@click.argument("name", required=False)
@click.option("--id", "profile_id", type=int, help="Lookup profile by id instead of name.")
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def delete_profile(name: Optional[str], profile_id: Optional[int], force: bool) -> None:
    _validate_identifier(name, profile_id)

    with utils.session_scope() as session:
        query = session.query(StrategyConfig)
        if profile_id is not None:
            profile = query.filter(StrategyConfig.id == profile_id).one_or_none()
        else:
            profile = query.filter(StrategyConfig.name == name).one_or_none()

        if profile is None:
            identifier = name if name else profile_id
            raise click.ClickException(f"Profile not found: {identifier}")

        profile_name = profile.name
        if not force:
            confirmed = click.confirm(f"Delete profile '{profile_name}'?", default=False)
            if not confirmed:
                click.echo("Deletion cancelled.")
                return

        session.delete(profile)

    click.echo(f"Profile '{profile_name}' deleted.")
