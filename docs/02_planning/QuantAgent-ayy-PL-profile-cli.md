# QuantAgent-ayy: Profile CLI Planning

**Issue:** QuantAgent-ayy  
**Type:** Feature  
**Priority:** P2  
**Estimated Effort:** 4-6 hours

---

## Task Breakdown

### T1: CLI Infrastructure Setup (1h)

**Goal:** Create CLI module structure and entry point

**Tasks:**
1. Create `quantagent/cli/` directory
2. Create `__init__.py` with version/metadata
3. Create `__main__.py` with Click app and main group
4. Create `utils.py` with `get_db_session()` context manager
5. Add Click and tabulate to `requirements.txt`

**Validation:**
- `python -m quantagent.cli --help` displays usage
- `python -m quantagent.cli --version` displays version
- DB session can be created from utils

**Depends on:** None

---

### T2: Profile List Command (0.5h)

**Goal:** Implement `profile list` with filtering and JSON output

**Tasks:**
1. Create `quantagent/cli/profile.py` with Click command group
2. Implement `list` command with:
   - `--kind` filter option
   - `--name-like` filter option
   - `--json` output flag
3. Add table formatter using tabulate
4. Add JSON serializer for datetime fields

**Validation:**
- List all profiles (empty DB, single, multiple)
- Filter by kind
- Filter by name pattern
- JSON output parseable by jq

**Depends on:** T1

---

### T3: Profile Show Command (0.5h)

**Goal:** Display single profile details

**Tasks:**
1. Implement `show` command with:
   - Positional `name` argument
   - `--id` option (alternative lookup)
   - `--json` output flag
2. Format json_config with indentation (text mode)
3. Handle not-found error

**Validation:**
- Show by name
- Show by id
- Not found returns exit code 1
- JSON output complete and parseable

**Depends on:** T1

---

### T4: Profile Create Command (1h)

**Goal:** Create profiles from JSON input

**Tasks:**
1. Implement `create` command with:
   - `--config` option for JSON string
   - stdin fallback if no --config
2. Parse and validate JSON input
3. Check required fields (name, kind, json_config)
4. Handle duplicate name error
5. Create profile with version=1

**Validation:**
- Create via --config flag
- Create via stdin
- Duplicate name error clear and actionable
- Invalid JSON parse error with location
- Missing fields error lists all missing

**Depends on:** T1

---

### T5: Profile Update Command (1.5h)

**Goal:** Update profiles with merge/replace logic

**Tasks:**
1. Implement `update` command with:
   - Positional `name` or `--id` lookup
   - `--config` option or stdin
   - `--replace` flag (default: merge)
   - `--keep-version` flag (default: increment)
2. Implement shallow merge logic for json_config
3. Implement replace logic (full overwrite)
4. Version increment (unless --keep-version)
5. Update updated_at timestamp

**Validation:**
- Merge behavior preserves unmodified fields
- Replace behavior overwrites json_config
- Version increments correctly
- --keep-version preserves version
- Not found returns error

**Depends on:** T1

---

### T6: Profile Delete Command (0.5h)

**Goal:** Delete profiles with confirmation

**Tasks:**
1. Implement `delete` command with:
   - Positional `name` or `--id` lookup
   - `--force` flag to skip confirmation
2. Add confirmation prompt (default: require confirmation)
3. Check BacktestRun usage (warning only, no FK block)
4. Delete profile and commit

**Validation:**
- Confirmation prompt displayed
- Force flag skips prompt
- Not found returns error
- Warning shown if BacktestRun references exist

**Depends on:** T1

---

### T7: Help Text & Documentation (0.5h)

**Goal:** Complete CLI help and examples

**Tasks:**
1. Add help text to each command
2. Add usage examples in help text
3. Add command descriptions
4. Test all `--help` outputs

**Validation:**
- `profile --help` lists all commands
- Each command help includes examples
- Help text clear and concise

**Depends on:** T2-T6

---

### T8: Unit Tests (1.5h)

**Goal:** Test coverage for CLI commands

**Tasks:**
1. Create `tests/cli/test_profile_cli.py`
2. Setup fixtures:
   - In-memory DB with sample profiles
   - Click CliRunner
3. Write tests for each command:
   - List (empty, multiple, filters, JSON)
   - Show (by name, by id, not found, JSON)
   - Create (success, duplicate, invalid JSON, missing fields)
   - Update (merge, replace, version, not found)
   - Delete (confirmation, force, not found)
4. Test error messages and exit codes

**Validation:**
- All commands covered
- Error cases tested
- JSON I/O round-trip validated
- Exit codes correct

**Depends on:** T2-T6

---

## Dependencies

```
T1 (CLI setup)
 ├── T2 (list)
 ├── T3 (show)
 ├── T4 (create)
 ├── T5 (update)
 └── T6 (delete)
      └── T7 (help)
           └── T8 (tests)
```

---

## Testing Strategy

### Development Testing
- Manual smoke tests after each task
- Test both text and JSON output modes
- Verify error messages are clear

### Pre-commit Testing
- Run full test suite (`pytest tests/cli/`)
- Manual integration test workflow:
  ```bash
  # Create profile
  echo '{"name":"test","kind":"portfolio","json_config":{}}' | \
    python -m quantagent.cli profile create
  
  # List profiles
  python -m quantagent.cli profile list
  
  # Show profile
  python -m quantagent.cli profile show test --json
  
  # Update profile
  python -m quantagent.cli profile update test \
    --config '{"json_config":{"universe":["BTC"]}}'
  
  # Delete profile
  python -m quantagent.cli profile delete test --force
  ```

### Acceptance Testing
- Run acceptance criteria from `docs/05_acceptance_tests/QuantAgent-ayy-AC-profile-cli.md`
- Verify exit codes, error messages, JSON validity
- Test against real database (dev environment)

---

## Rollout

### Phase 1: Development
- Complete T1-T6 (core functionality)
- Manual testing in dev environment

### Phase 2: Testing
- Complete T7-T8 (help text, unit tests)
- Run full acceptance suite

### Phase 3: Integration
- Update README with CLI usage examples
- Document in repo (add to `docs/` index if needed)
- Human review and approval

### Phase 4: Deployment
- Merge to main via feature branch
- No deployment needed (CLI available via `python -m`)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Click version incompatibility | Medium | Pin Click>=8.0, test with min version |
| DB connection issues in CLI | High | Reuse existing quantagent.database setup |
| JSON parsing edge cases | Low | Use standard json library, handle errors clearly |
| FK constraint on delete | Low | Warn only, defer to future issue if needed |

---

## Open Questions

None (all decisions made in design phase).

---

## Success Metrics

- CLI accessible via `python -m quantagent.cli profile`
- All CRUD operations functional
- JSON I/O validated
- Version tracking verified
- Error messages tested
- Zero breaking changes to existing code

---

## Follow-up Issues (Out of Scope)

- REST API endpoints for profile management
- Profile validation against assembler rules
- Profile templates and wizards
- Profile export/import utilities
- Profile history/audit log
