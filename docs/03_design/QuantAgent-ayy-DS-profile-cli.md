# QuantAgent-ayy: Profile CLI Design

**Issue:** QuantAgent-ayy  
**Type:** Feature  
**Status:** Planning

---

## Overview

Implement a CLI module for StrategyConfig CRUD operations using Click framework, following Python standard practices for CLI tools.

---

## Architecture

### Module Structure

```
quantagent/
├── cli/
│   ├── __init__.py
│   ├── __main__.py          # Entry point: python -m quantagent.cli
│   ├── profile.py           # Profile command group
│   └── utils.py             # Shared utilities (db session, formatters)
```

### Entry Point

`quantagent/cli/__main__.py`:
- Main CLI entry point
- Command groups: `profile` (extensible for future commands)
- Global options: `--verbose`, `--quiet`

---

## Technology Choices

### CLI Framework: Click

**Rationale:**
- Standard in Python ecosystem
- Automatic help generation
- Nested command groups
- Built-in validation

**Alternative (not chosen):** Typer
- Reason: Click more mature, better documented, simpler for CRUD operations

---

## Component Design

### Database Session Management

**Approach:** Context manager per command  
**Location:** `quantagent/cli/utils.py`

```python
@contextmanager
def get_db_session():
    """Provide database session for CLI commands."""
    # Use existing quantagent.database setup
    # Respect DATABASE_URL env var
```

**Rationale:** Existing database setup reused, no duplication.

---

### Output Formatting

**Text Mode (default):**
- Tables: use `tabulate` library
- Single records: YAML-like key-value pairs
- Colorized output: `click.style()` for success/error

**JSON Mode (`--json`):**
- Use `json.dumps()` with `default=str` for datetime serialization
- Ensure output is valid, parseable JSON

---

### Command Specifications

#### `profile list`

**Options:**
- `--kind [portfolio|risk|combined]`
- `--name-like TEXT`
- `--json`

**Query:**
```python
query = session.query(StrategyConfig)
if kind:
    query = query.filter(StrategyConfig.kind == kind)
if name_like:
    query = query.filter(StrategyConfig.name.like(f"%{name_like}%"))
return query.order_by(StrategyConfig.created_at.desc()).all()
```

---

#### `profile show`

**Arguments:**
- `name` (positional) OR `--id INTEGER`

**Logic:**
- Lookup by name or id (mutually exclusive)
- If not found: error + exit 1
- If found: display all fields

**JSON config display (text mode):**
- Pretty-print with `json.dumps(indent=2)`

---

#### `profile create`

**Options:**
- `--config TEXT` (JSON string) OR stdin if not provided

**Input parsing:**
```python
if config_text:
    data = json.loads(config_text)
else:
    data = json.loads(sys.stdin.read())

# Validate required fields
assert "name" in data
assert "kind" in data
assert "json_config" in data
```

**Creation:**
```python
profile = StrategyConfig(
    name=data["name"],
    kind=data["kind"],
    json_config=data["json_config"],
    version=data.get("version", 1),
)
session.add(profile)
session.commit()
```

**Error handling:**
- Duplicate name: catch `IntegrityError`, display friendly message
- Invalid JSON: catch `json.JSONDecodeError`, show error location

---

#### `profile update`

**Arguments:**
- `name` (positional) OR `--id INTEGER`

**Options:**
- `--config TEXT` (JSON string) OR stdin
- `--replace` (default: merge)
- `--keep-version` (default: increment)

**Merge logic (default):**
```python
profile = session.query(StrategyConfig).filter(...).one()
update_data = json.loads(config_text)

if "json_config" in update_data:
    # Shallow merge
    profile.json_config = {**profile.json_config, **update_data["json_config"]}

if not keep_version:
    profile.version += 1

session.commit()
```

**Replace logic (`--replace`):**
```python
if "json_config" in update_data:
    profile.json_config = update_data["json_config"]
```

---

#### `profile delete`

**Arguments:**
- `name` (positional) OR `--id INTEGER`

**Options:**
- `--force` (skip confirmation)

**Confirmation:**
```python
if not force:
    click.confirm(f"Delete profile '{profile.name}'?", abort=True)

session.delete(profile)
session.commit()
click.echo(f"Profile '{profile.name}' deleted.")
```

**FK check (warning only):**
```python
backtest_count = session.query(BacktestRun).filter(
    BacktestRun.config_snapshot.contains({"name": profile.name})
).count()

if backtest_count > 0:
    click.echo(f"Warning: Profile used by {backtest_count} backtest run(s).", err=True)
```

---

## Error Handling

### General Strategy

- **User errors:** Exit code 1, clear message, suggestion if applicable
- **System errors:** Exit code 2, stack trace if `--verbose`
- **Validation errors:** Exit code 1, list all validation failures

### Specific Cases

| Error | Message | Exit Code |
|-------|---------|-----------|
| Profile not found | `Profile not found: {name}` | 1 |
| Duplicate name | `Profile with name '{name}' already exists. Use 'update' or choose different name.` | 1 |
| Invalid JSON | `JSON parse error at line X, column Y: {error}` | 1 |
| Missing required field | `Missing required fields: name, kind` | 1 |
| DB connection failed | `Database connection failed: {error}` | 2 |

---

## Testing Strategy

### Unit Tests

**Location:** `tests/cli/test_profile_cli.py`

**Fixtures:**
- In-memory SQLite database
- Sample StrategyConfig records

**Test coverage:**
- Each command with valid input
- Each error case (not found, duplicate, invalid JSON)
- JSON I/O round-trip
- Version increment logic
- Merge vs replace behavior

### Integration Tests

**Manual validation:**
- Real database (dev environment)
- Multi-step workflows (create → update → show → delete)
- Shell pipe compatibility (`echo ... | python -m ...`)

---

## Dependencies

**New:**
- `click>=8.0` (add to `requirements.txt`)
- `tabulate>=0.9` (for table formatting)

**Existing:**
- `sqlalchemy` (database)
- `python-json-logger` (logging, if needed)

---

## Migration & Compatibility

- No database schema changes
- No breaking changes to StrategyConfig model
- Streamlit UI continues to work unchanged
- CLI and UI can coexist (use same database)

---

## Security & Validation

- **SQL injection:** Prevented by SQLAlchemy ORM
- **JSON validation:** Basic type checks only; defer business logic to StrategyAssembler
- **File system access:** None (stdin/stdout only)
- **Environment variables:** Respect existing `DATABASE_URL`

---

## Future Extensions (Out of Scope)

- Profile templates (`profile init --template portfolio`)
- Profile validation against assembler rules
- Profile export/import with metadata
- REST API endpoints (separate issue)
- Profile history/audit log

---

## References

- **Model:** `quantagent/models.py::StrategyConfig`
- **Click docs:** https://click.palletsprojects.com/
- **Related design:** `docs/03_design/strategy_assembler_architecture.md`
