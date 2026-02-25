# Profile CLI Guide

The Profile CLI lets you create, inspect, update, and delete strategy profiles directly from the terminal. It mirrors everything you can do in the Dashboard → Configuration tab, so you can automate workflows or manage profiles from a headless server.

> Implemented in **QuantAgent-ayy**. Source: `quantagent/cli/*`, tests in `tests/test_profile_cli.py`.

---

## Prerequisites

1. **Activate the QuantAgent environment**
   ```bash
   cd ~/repos/projects/QuantAgent
   poetry shell    # or source your virtualenv
   ```
2. **Database connection**
   - Local dev reuses the settings in `quantagent/settings.py` (PostgreSQL by default).
   - To point at another database, set `DATABASE_URL` before running commands:
     ```bash
     export DATABASE_URL=postgresql://user:pass@host:5432/quantagent
     ```
   - The CLI automatically reloads engines when this variable changes, so no restart is needed.
3. **Command entry point**
   - Run everything through `python -m quantagent.cli …`
   - Add `profile` subcommands for day-to-day use:
     ```bash
     python -m quantagent.cli profile --help
     ```

---

## Quick Reference

| Command | What it does |
|---------|---------------|
| `profile list` | List saved profiles (table or JSON) |
| `profile show <name>` | Print a single profile, including `json_config` |
| `profile create` | Insert a new profile from JSON (`--config` or stdin) |
| `profile update <name>` | Merge or replace fields on an existing profile |
| `profile delete <name>` | Remove a profile (with confirmation unless `--force`) |

All commands support `--id <number>` as an alternative to the profile name when you need exact targeting.

---

## Listing Profiles

```bash
python -m quantagent.cli profile list
```

- **Filters**
  - `--kind portfolio|risk|combined`
  - `--name-like value` (automatically wraps `%value%` if you omit the `%` wildcards)
- **Machine-friendly output**: add `--json` and pipe to `jq`
  ```bash
  python -m quantagent.cli profile list --json | jq '.[].name'
  ```
- Table headers follow the same naming as the database columns (`id`, `name`, `kind`, `version`, `created_at`, `updated_at`).

---

## Showing a Profile

```bash
python -m quantagent.cli profile show swing-long
python -m quantagent.cli profile show --id 42 --json
```

- Accepts either the profile name (`swing-long`) or `--id`.
- Prints metadata plus the exact `json_config` block that the Dashboard uses.
- `--json` returns a single JSON object, perfect for scripting or version control diffs.

---

## Creating Profiles

Provide a JSON payload either inline (`--config '{...}'`) or via stdin:

```bash
cat <<'EOF' | python -m quantagent.cli profile create
{
  "name": "balanced-multi-asset",
  "kind": "combined",
  "json_config": {
    "universe": ["BTC", "SPX", "CL"],
    "base_position_pct": 0.05,
    "max_position_pct": 0.10,
    "max_daily_loss_pct": 0.05,
    "slippage_pct": 0.01
  }
}
EOF
```

Rules:
- Required fields: `name`, `kind`, `json_config`
- Optional: `version` (defaults to `1`)
- Duplicate names throw a human-readable error so you never overwrite silently.

---

## Updating Profiles

```bash
# Merge fields (default behavior)
python -m quantagent.cli profile update swing-long --config '{"json_config": {"max_daily_loss_pct": 0.04}}'

# Replace the json_config entirely and keep the current version number
python -m quantagent.cli profile update swing-long \
  --config @new_profile.json \
  --replace \
  --keep-version
```

Options:
- `--config` behaves the same as in `create` (inline or `@file.json`).
- `--replace` swaps the full `json_config`. Without it, QuantAgent merges the incoming keys into the existing JSON.
- `--keep-version` prevents the automatic version bump. Otherwise, the CLI increments `version` unless you explicitly send a new number in the payload.

Useful when pairing with Git scripts: store canonical JSON in your repo and push updates in CI/CD.

---

## Deleting Profiles

```bash
python -m quantagent.cli profile delete swing-long
python -m quantagent.cli profile delete --id 42 --force
```

- Interactive confirmation protects you from accidental deletions.
- Use `--force` when running in automation or when stdin is not attached to a TTY.
- Deletion removes the profile immediately from the database; there is no recycle bin, so keep backups if needed.

---

## Automation Tips

- **Combine with CI**: Add a pipeline step that runs `profile list --json` to detect drifts between branches.
- **Audit changes**: Pipe `profile show --json` into `git diff --no-index` to compare versions over time.
- **Headless deployments**: Set `DATABASE_URL` to your staging/production database and run the CLI from your server—no web UI required.
- **Consistency with Dashboard**: After CLI updates, refresh the Configuration tab to see the new profile. The same APIs power both surfaces, so changes are instantaneous.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Database connection failed` | Wrong `DATABASE_URL` or DB down | Export the correct URL and rerun. Ensure PostgreSQL accepts connections. |
| `Profile not found` | Typo in name or wrong environment | Use `profile list --json` first, or target by `--id`. |
| `Missing required field(s)` | Payload lacks `name`, `kind`, or `json_config` | Double-check your JSON, especially when using shell heredocs. |
| `Invalid JSON` | Trailing commas or comments in payload | Validate with `jq . file.json` before calling the CLI. |

---

## Related Reading

- Strategy concepts: [Strategy Configuration Guide](strategy-configuration.md)
- QuantAgent-ayy docs:
  - [Requirements](../01_requirements/QuantAgent-ayy-RQ-profile-cli.md)
  - [Design](../03_design/QuantAgent-ayy-DS-profile-cli.md)
  - [Acceptance Tests](../05_acceptance_tests/QuantAgent-ayy-AC-profile-cli.md)
- Test suite: [`tests/test_profile_cli.py`](../../tests/test_profile_cli.py)

Use the CLI whenever you need deterministic, scriptable control over profiles—especially helpful for syncing environments or onboarding teammates without exposing the full dashboard.
