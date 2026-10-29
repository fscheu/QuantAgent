# `quantagent profile` command reference

```text
Usage: quantagent profile [OPTIONS] COMMAND [ARGS]...

  Manage StrategyConfig profiles (list, show, create, update, delete).

Options:
  --help  Show this message and exit.

Commands:
  create  Create a new profile from JSON input.
  delete  Delete a profile by name or id.
  list    List profiles with optional filtering.
  show    Show details for a profile by name or id.
  update  Update fields on an existing profile.
```

## `quantagent profile create`

```text
Usage: quantagent profile create [OPTIONS]

  Create a new profile from JSON input.

Options:
  --config TEXT  Inline JSON payload. If omitted, read from stdin.
  --help         Show this message and exit.
```

Options and defaults:

- `--config` — default: `None`

## `quantagent profile delete`

```text
Usage: quantagent profile delete [OPTIONS] [NAME]

  Delete a profile by name or id.

Options:
  --id INTEGER  Lookup profile by id instead of name.
  --force       Skip confirmation prompt.
  --help        Show this message and exit.
```

Options and defaults:

- `--id` — default: `None`
- `--force` — default: `False`

## `quantagent profile list`

```text
Usage: quantagent profile list [OPTIONS]

  List profiles with optional filtering.

Options:
  --kind [portfolio|risk|combined]
                                  Filter by profile kind.
  --name-like TEXT                Filter by SQL LIKE pattern (e.g.,
                                  %portfolio%).
  --json                          Output as JSON instead of a table.
  --help                          Show this message and exit.
```

Options and defaults:

- `--kind` — default: `None`
- `--name-like` — default: `None`
- `--json` — default: `False`

## `quantagent profile show`

```text
Usage: quantagent profile show [OPTIONS] [NAME]

  Show details for a profile by name or id.

Options:
  --id INTEGER  Lookup profile by numeric id.
  --json        Return data as JSON.
  --help        Show this message and exit.
```

Options and defaults:

- `--id` — default: `None`
- `--json` — default: `False`

## `quantagent profile update`

```text
Usage: quantagent profile update [OPTIONS] [NAME]

  Update fields on an existing profile.

Options:
  --id INTEGER    Lookup profile by id instead of name.
  --config TEXT   JSON payload with fields to update.
  --replace       Replace json_config instead of merging.
  --keep-version  Do not increment the version number.
  --help          Show this message and exit.
```

Options and defaults:

- `--id` — default: `None`
- `--config` — default: `None`
- `--replace` — default: `False`
- `--keep-version` — default: `False`
