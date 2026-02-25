# QuantAgent-ayy: Profile CLI/API Requirements

**Issue:** QuantAgent-ayy  
**Type:** Feature  
**Priority:** P2  
**Labels:** cli, configuration, dx

---

## Objective

Provide CLI commands and API endpoints for CRUD operations on StrategyConfig profiles, eliminating the need for direct database manipulation or UI-only access for profile management.

---

## Scope

### In Scope

- CLI commands for profile management:
  - `list` — list all profiles with filtering
  - `show` — display profile details
  - `create` — create new profile from JSON
  - `update` — modify existing profile
  - `delete` — remove profile

- Support for all profile kinds:
  - `portfolio` — universe, cash, position sizing, slippage
  - `risk` — max loss, max position constraints
  - `combined` — merged portfolio + risk

- JSON input/output format for programmatic use

- Version tracking on updates (increment version field)

### Out of Scope

- Profile validation against business rules (defer to StrategyAssembler)
- Profile templates or wizards
- REST API endpoints (CLI only for MVP)
- Profile history/rollback (version field tracks updates only)
- Profile import/export utilities (use shell redirection)

---

## Current State

- StrategyConfig model exists: `name`, `kind`, `json_config`, `version`, timestamps
- Profiles managed via:
  - Direct database inserts (developer workflow)
  - Streamlit UI (non-developer workflow)
- No CLI infrastructure in `quantagent/` package

---

## Requirements

### FR1: List Profiles

- Display all profiles in tabular format
- Columns: id, name, kind, version, created_at, updated_at
- Support filtering:
  - `--kind portfolio|risk|combined`
  - `--name-like <pattern>` (SQL LIKE)
- JSON output option: `--json`

### FR2: Show Profile

- Display single profile by name or id
- Output: all fields including full `json_config`
- Default: human-readable (YAML-like)
- Option: `--json` for machine parsing

### FR3: Create Profile

- Input: JSON string via `--config` or stdin
- Required fields: `name`, `kind`, `json_config`
- Optional: `version` (default: 1)
- Validation:
  - Unique name constraint
  - Valid kind enum
  - json_config is valid JSON object
- Return: created profile details

### FR4: Update Profile

- Identify profile by name or id
- Input: partial or full JSON config via `--config` or stdin
- Behavior:
  - Merge with existing `json_config` (shallow merge)
  - Increment version
  - Update `updated_at` timestamp
- Options:
  - `--replace` — full replacement instead of merge
  - `--increment-version` / `--keep-version` (default: increment)
- Return: updated profile details

### FR5: Delete Profile

- Identify profile by name or id
- Confirmation prompt unless `--force`
- Return: deletion confirmation message

---

## Constraints

- CLI must use existing database connection (respect `DATABASE_URL` env var)
- No breaking changes to StrategyConfig model
- CLI framework: prefer Click or Typer (lightweight, standard)
- Error messages must be clear and actionable

---

## Edge Cases

- Creating profile with duplicate name → error with suggestion
- Updating non-existent profile → error
- Deleting profile referenced by BacktestRun → warn but allow (FK constraint decision TBD)
- Invalid JSON input → parse error with line/column info
- Empty database → list returns empty, no error

---

## Definition of Done

- CLI commands executable via `python -m quantagent.cli profile <command>`
- All commands support `--help`
- JSON input/output tested with real profiles
- Version increment verified on updates
- Error messages tested for common mistakes
- Documentation: usage examples in CLI help text

---

## References

- **Architecture:** `docs/03_design/strategy_assembler_architecture.md`
- **Model:** `quantagent/models.py::StrategyConfig`
- **Related Issues:** N/A
