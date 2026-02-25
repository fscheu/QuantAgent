# QuantAgent-ayy: Profile CLI Acceptance Criteria

**Issue:** QuantAgent-ayy  
**Type:** Feature  
**Status:** Planning

---

## AC1: List Profiles

**Given:** Multiple profiles exist in database  
**When:** User runs `python -m quantagent.cli profile list`  
**Then:**
- Display table with columns: id, name, kind, version, created_at, updated_at
- All profiles visible
- Exit code 0

**Given:** User runs `python -m quantagent.cli profile list --kind portfolio`  
**When:** Database contains portfolio, risk, and combined profiles  
**Then:**
- Only portfolio profiles displayed
- Exit code 0

**Given:** User runs `python -m quantagent.cli profile list --json`  
**When:** Profiles exist  
**Then:**
- Output valid JSON array
- Each object contains all StrategyConfig fields
- Parseable by `jq` or `json.loads()`

**Given:** Empty database  
**When:** User runs `python -m quantagent.cli profile list`  
**Then:**
- Message: "No profiles found"
- Exit code 0 (not an error)

---

## AC2: Show Profile

**Given:** Profile named "default-portfolio" exists  
**When:** User runs `python -m quantagent.cli profile show default-portfolio`  
**Then:**
- Display name, kind, version, timestamps
- Display full `json_config` (formatted)
- Exit code 0

**Given:** Profile with id=5 exists  
**When:** User runs `python -m quantagent.cli profile show --id 5`  
**Then:**
- Same output as by-name lookup
- Exit code 0

**Given:** Profile does not exist  
**When:** User runs `python -m quantagent.cli profile show nonexistent`  
**Then:**
- Error: "Profile not found: nonexistent"
- Exit code 1

**Given:** Profile exists  
**When:** User runs `python -m quantagent.cli profile show <name> --json`  
**Then:**
- Output valid JSON object
- Parseable and complete

---

## AC3: Create Profile

**Given:** Valid JSON config via stdin  
```bash
echo '{"name": "test-portfolio", "kind": "portfolio", "json_config": {"universe": ["BTC"], "base_position_pct": 0.05}}' | \
  python -m quantagent.cli profile create
```
**Then:**
- Profile created in database
- Version = 1
- Confirmation message with id
- Exit code 0

**Given:** Valid JSON config via --config  
**When:** User runs:
```bash
python -m quantagent.cli profile create --config '{"name": "test-risk", "kind": "risk", "json_config": {"max_daily_loss_pct": 0.05}}'
```
**Then:**
- Profile created
- Exit code 0

**Given:** Duplicate name  
**When:** User tries to create profile with existing name  
**Then:**
- Error: "Profile with name 'X' already exists"
- Suggestion: "Use 'update' command or choose different name"
- Exit code 1

**Given:** Invalid JSON  
**When:** User provides malformed JSON  
**Then:**
- Parse error with line/column info
- Exit code 1

**Given:** Missing required field  
**When:** JSON lacks `name`, `kind`, or `json_config`  
**Then:**
- Validation error listing missing fields
- Exit code 1

---

## AC4: Update Profile

**Given:** Profile "default-portfolio" exists with version=1  
**When:** User runs:
```bash
python -m quantagent.cli profile update default-portfolio --config '{"json_config": {"slippage_pct": 0.02}}'
```
**Then:**
- `json_config.slippage_pct` updated to 0.02
- Other fields in `json_config` unchanged (merge behavior)
- Version incremented to 2
- `updated_at` timestamp updated
- Exit code 0

**Given:** Profile exists  
**When:** User runs with `--replace` flag:
```bash
python -m quantagent.cli profile update <name> --replace --config '{"json_config": {"universe": ["ETH"]}}'
```
**Then:**
- `json_config` fully replaced (no merge)
- Version incremented
- Exit code 0

**Given:** Non-existent profile  
**When:** User tries to update  
**Then:**
- Error: "Profile not found: X"
- Exit code 1

**Given:** Update with `--keep-version`  
**When:** User runs:
```bash
python -m quantagent.cli profile update <name> --keep-version --config '...'
```
**Then:**
- Version unchanged
- Exit code 0

---

## AC5: Delete Profile

**Given:** Profile "test-profile" exists  
**When:** User runs:
```bash
python -m quantagent.cli profile delete test-profile
```
**Then:**
- Confirmation prompt: "Delete profile 'test-profile'? [y/N]"
- If user confirms (y): profile deleted, success message, exit 0
- If user declines (n): no deletion, cancelled message, exit 0

**Given:** Profile exists  
**When:** User runs with `--force`:
```bash
python -m quantagent.cli profile delete test-profile --force
```
**Then:**
- No confirmation prompt
- Profile deleted immediately
- Exit code 0

**Given:** Non-existent profile  
**When:** User tries to delete  
**Then:**
- Error: "Profile not found: X"
- Exit code 1

**Given:** Profile referenced by BacktestRun  
**When:** User tries to delete (with confirmation)  
**Then:**
- Warning: "Profile used by N backtest run(s)"
- Deletion proceeds (no FK constraint in MVP)
- Exit code 0

---

## AC6: CLI Usability

**Given:** User runs `python -m quantagent.cli profile --help`  
**Then:**
- Display list of available commands
- Brief description of each
- Exit code 0

**Given:** User runs `python -m quantagent.cli profile <command> --help`  
**Then:**
- Display command-specific usage
- List all options and arguments
- Include examples
- Exit code 0

**Given:** Invalid command  
**When:** User runs `python -m quantagent.cli profile invalid`  
**Then:**
- Error: "Unknown command: invalid"
- Suggestion: "Run 'profile --help' for available commands"
- Exit code 1

---

## Testing Checklist

- [ ] List command: empty DB, single profile, multiple profiles, filtering
- [ ] Show command: by name, by id, not found, JSON output
- [ ] Create command: stdin input, --config flag, duplicate name, invalid JSON
- [ ] Update command: merge, replace, keep-version, increment-version, not found
- [ ] Delete command: confirmation, force flag, not found
- [ ] Help text: main help, command-specific help
- [ ] Error messages: clear, actionable, correct exit codes
- [ ] JSON I/O: parseable, complete, correct types
- [ ] Version tracking: increments on update, preservable with flag
