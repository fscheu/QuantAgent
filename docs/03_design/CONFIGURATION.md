# Configuration Management

## Overview

QuantAgent uses environment-based configuration through `.env` files for managing database connections, LLM API keys, and model settings. This approach provides:

- **Single source of truth**: All configuration in one file
- **Environment separation**: Different settings for dev/staging/production
- **Security**: Credentials never committed to version control
- **Runtime flexibility**: Update API keys without code changes
- **Type safety**: Centralized settings module with typed constants

## Architecture

### Configuration Flow

```
.env file → settings.py (loads & validates) → Application modules
                ↓
         update_env_file()
                ↓
         .env file (persistence)
```

### Core Components

1. **`.env` file**: Environment variables (gitignored)
2. **`.env.example`**: Template with documentation
3. **`quantagent/settings.py`**: Centralized configuration module
4. **`quantagent/default_settings.py`**: Model defaults (fallback values)

## Configuration Variables

### Database Configuration

```env
DATABASE_URL=postgresql://user:password@localhost:5432/quantagent
```

**Purpose**: PostgreSQL connection string for SQLAlchemy ORM

**Used by**:
- `quantagent/database.py` - Database engine initialization
- `quantagent/trading_graph.py:_setup_checkpointer()` - LangGraph checkpointing
- `alembic/env.py` - Database migrations

**Validation**: Raises `ValueError` if not set when database module is imported

### LLM API Keys

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DASHSCOPE_API_KEY=sk-...
```

**Purpose**: Authentication for LLM provider APIs

**Used by**:
- `quantagent/trading_graph.py:_get_api_key()` - LLM client initialization
- `quantagent/trading_graph.py:_create_llm()` - Provider-specific client creation
- `apps/flask/web_interface.py:validate_api_key()` - API key validation

**Validation**: Raises `ValueError` with provider-specific message if key missing when LLM is created

### LLM Provider Configuration

```env
AGENT_LLM_PROVIDER=anthropic
GRAPH_LLM_PROVIDER=anthropic
```

**Purpose**: Select which LLM provider to use for each agent type

**Supported values**: `openai`, `anthropic`, `qwen`

**Used by**:
- `quantagent/trading_graph.py:__init__()` - Initialize agent and graph LLMs
- `quantagent/trading_graph.py:refresh_llms()` - Recreate LLMs after config changes
- `apps/flask/web_interface.py:update_provider()` - Runtime provider switching

**Default**: `openai` (if not specified)

### LLM Model Configuration

```env
AGENT_LLM_MODEL=claude-haiku-4-5-20251001
GRAPH_LLM_MODEL=claude-haiku-4-5-20251001
```

**Purpose**: Specify exact model version for each agent type

**Used by**:
- `quantagent/trading_graph.py:__init__()` - Model selection
- `quantagent/settings.py:get_default_model()` - Provider-based defaults

**Defaults** (if not specified):
- OpenAI: `gpt-4o-mini` (agent), `gpt-4o` (graph)
- Anthropic: `claude-haiku-4-5-20251001` (both)
- Qwen: `qwen3-max` (agent), `qwen3-vl-plus` (graph)

### LLM Temperature

```env
AGENT_LLM_TEMPERATURE=0.1
GRAPH_LLM_TEMPERATURE=0.1
```

**Purpose**: Control LLM output randomness (0.0 = deterministic, 1.0+ = creative)

**Used by**:
- `quantagent/trading_graph.py:__init__()` - Set temperature for each LLM
- All agent nodes via LangChain LLM instances

**Default**: `0.1` (professional, deterministic outputs for trading decisions)

**Important**: Do not increase above 0.1 without justification - higher values risk unreliable trading analysis

## Configuration Module (`quantagent/settings.py`)

### Loading Mechanism

```python
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
```

- **When loaded**: On first import of `quantagent.settings`
- **Idempotent**: `load_dotenv()` only loads once, even if imported multiple times
- **Path**: Looks for `.env` in project root (parent of `quantagent/`)

### Exported Constants

All configuration is exposed as module-level constants:

```python
from quantagent import settings

# Database
settings.DATABASE_URL

# API Keys
settings.OPENAI_API_KEY
settings.ANTHROPIC_API_KEY
settings.DASHSCOPE_API_KEY

# Providers
settings.AGENT_LLM_PROVIDER
settings.GRAPH_LLM_PROVIDER

# Models
settings.AGENT_LLM_MODEL
settings.GRAPH_LLM_MODEL

# Temperature
settings.AGENT_LLM_TEMPERATURE
settings.GRAPH_LLM_TEMPERATURE
```

### Persistence Function

`settings.update_env_file(key: str, value: str)` updates both runtime and `.env` file:

**Used by**:
- `quantagent/trading_graph.py:update_api_key()` - Persist API key changes
- `apps/flask/web_interface.py:update_provider()` - Persist provider changes
- `apps/flask/web_interface.py:update_api_key()` - Web interface API key updates

**Behavior**:
1. Reads existing `.env` file
2. Updates or appends key-value pair
3. Writes back to `.env`
4. Updates `os.environ` for runtime availability

## Configuration Usage by Module

### `quantagent/trading_graph.py`

**Configuration read on initialization**:
- Provider selection → `_create_llm()`
- Model selection → `_create_llm()`
- Temperature → `_create_llm()`
- Database URL → `_setup_checkpointer()` (if checkpointing enabled)

**Configuration updates**:
- `update_api_key()` → Persists to `.env`, updates runtime, refreshes LLMs
- `refresh_llms()` → Recreates LLM instances with current config values

### `quantagent/database.py`

**Configuration read on import**:
- `settings.DATABASE_URL` → Engine creation
- Validation raises error if not set

**Connection pool behavior**:
- SQLite: `StaticPool` (single connection)
- PostgreSQL/MySQL: Connection pooling with `pool_pre_ping=True`

### `apps/flask/web_interface.py`

**Configuration read**:
- `WebTradingAnalyzer.__init__()` → Initializes `TradingGraph` (reads all config)
- API endpoints → Read current config for status/validation

**Configuration updates**:
- `/api/update-api-key` → Calls `trading_graph.update_api_key()` → Persists to `.env`
- `/api/update-provider` → Updates provider + models → Persists to `.env`
- `/api/get-api-key-status` → Reads current keys from settings module

### `alembic/env.py`

**Configuration read**:
- `settings.DATABASE_URL` → Database migrations

**Note**: Must import `quantagent.settings` to trigger `.env` loading before accessing `DATABASE_URL`

## Environment-Specific Configuration

### Development Environment

**File**: `.env` (local, gitignored)

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantagent_dev
OPENAI_API_KEY=sk-dev-key
ANTHROPIC_API_KEY=sk-ant-dev-key
AGENT_LLM_PROVIDER=anthropic
GRAPH_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-haiku-4-5-20251001
GRAPH_LLM_MODEL=claude-haiku-4-5-20251001
```

**Characteristics**:
- Local PostgreSQL or Docker container
- Development-tier API keys (if available)
- Can use SQLite for quick testing: `DATABASE_URL=sqlite:///./quantagent_dev.db`

**Setup**:
```bash
cp .env.example .env
# Edit .env with your local credentials
docker-compose up -d  # Start PostgreSQL
python -m alembic upgrade head
```

### Staging/Testing Environment

**File**: `.env.staging` (deployed separately)

```env
DATABASE_URL=postgresql://user:password@staging-db.internal:5432/quantagent_staging
OPENAI_API_KEY=sk-staging-key
ANTHROPIC_API_KEY=sk-ant-staging-key
AGENT_LLM_PROVIDER=anthropic
GRAPH_LLM_PROVIDER=anthropic
```

**Characteristics**:
- Staging database instance
- Staging-tier API keys with rate limits
- Same configuration structure as production

**Deployment**:
```bash
# Copy staging env file
cp .env.staging .env
# Or use environment variable override
export DATABASE_URL=postgresql://...
```

### Production Environment

**File**: `.env.prod` (secure deployment)

```env
DATABASE_URL=postgresql://quantagent:${DB_PASSWORD}@prod-db.internal:5432/quantagent_prod
OPENAI_API_KEY=${OPENAI_KEY_SECRET}
ANTHROPIC_API_KEY=${ANTHROPIC_KEY_SECRET}
AGENT_LLM_PROVIDER=anthropic
GRAPH_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-haiku-4-5-20251001
GRAPH_LLM_MODEL=claude-haiku-4-5-20251001
AGENT_LLM_TEMPERATURE=0.1
GRAPH_LLM_TEMPERATURE=0.1
```

**Characteristics**:
- Production database with connection pooling
- Production API keys with monitoring
- Fixed model versions (no auto-updates)
- Temperature locked at 0.1

**Security considerations**:
- Use secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
- Inject secrets as environment variables: `export DB_PASSWORD=$(aws secretsmanager get-secret-value ...)`
- Never commit `.env.prod` to version control
- Restrict file permissions: `chmod 600 .env`

## Configuration Validation

### At Import Time

**`quantagent/database.py`**:
```python
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env file...")
```

Fails fast if database configuration missing.

### At LLM Creation Time

**`quantagent/trading_graph.py:_get_api_key()`**:
```python
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file...")
```

Only validates API key when LLM is actually created (lazy validation).

### Via Web Interface

**`apps/flask/web_interface.py:validate_api_key()`**:

Makes test API call to verify key validity:
- Returns `{"valid": True}` if successful
- Returns `{"valid": False, "error": "..."}` with specific error message

**Endpoint**: `POST /api/validate-api-key`

## Dynamic Configuration Updates

### Web Interface Updates

**API Key Update**:
1. User submits new key via web interface
2. `POST /api/update-api-key` → `analyzer.trading_graph.update_api_key()`
3. `settings.update_env_file()` persists to `.env`
4. Config module attributes updated (`settings.OPENAI_API_KEY = new_key`)
5. `refresh_llms()` recreates LLM instances with new key

**Provider Update**:
1. User selects new provider (e.g., switch from OpenAI to Anthropic)
2. `POST /api/update-provider` endpoint
3. Updates provider and model variables in settings module
4. Persists all changes to `.env`
5. `refresh_llms()` recreates LLM instances with new provider

**Key insight**: Changes persist across application restarts because `.env` file is updated.

### Programmatic Updates

```python
from quantagent import settings
from quantagent.trading_graph import TradingGraph

# Update API key programmatically
tg = TradingGraph()
tg.update_api_key("sk-new-key", provider="openai")
# This persists to .env automatically
```

## Best Practices

### Security

1. **Never commit `.env` files**: Already in `.gitignore`
2. **Use `.env.example` as template**: Safe to commit, documents required variables
3. **Restrict file permissions**: `chmod 600 .env` on production servers
4. **Use secrets management**: Inject secrets as environment variables in production
5. **Rotate API keys regularly**: Update via web interface or `update_api_key()`

### Configuration Management

1. **Copy `.env.example` first**: `cp .env.example .env`
2. **Validate after changes**: Use `/api/validate-api-key` endpoint
3. **Document custom variables**: Add comments in `.env` file
4. **One `.env` per environment**: `.env` (dev), `.env.staging`, `.env.prod`
5. **Version control `.env.example`**: Keep template up to date

### Model Selection

1. **Pin versions in production**: `AGENT_LLM_MODEL=claude-haiku-4-5-20251001`
2. **Test new models in staging first**: Update `.env.staging` before `.env.prod`
3. **Keep temperature at 0.1 for trading**: Deterministic outputs critical for financial decisions
4. **Use vision-capable models for pattern/trend agents**: Required for chart analysis

### Database Configuration

1. **Use PostgreSQL for production**: SQLite only for development/testing
2. **Enable SSL for remote databases**: `?sslmode=require` in DATABASE_URL
3. **Use connection pooling**: Automatically enabled for PostgreSQL
4. **Separate databases per environment**: `quantagent_dev`, `quantagent_staging`, `quantagent_prod`

## Troubleshooting

### "DATABASE_URL not set in .env file"

**Cause**: `.env` file missing or DATABASE_URL not defined

**Solution**:
```bash
cp .env.example .env
# Edit .env and set DATABASE_URL
```

### "OPENAI_API_KEY not found in .env file"

**Cause**: API key not set or provider mismatch

**Solution**:
1. Check `.env` has correct key: `OPENAI_API_KEY=sk-...`
2. Verify provider matches: `AGENT_LLM_PROVIDER=openai`
3. Restart application to reload `.env`

### Config changes not taking effect

**Cause**: Application needs restart to reload `.env`

**Solution**:
- For web interface: Restart Flask app (`python apps/flask/web_interface.py`)
- For scripts: Re-run script
- For API key changes: Use web interface `/api/update-api-key` (no restart needed)

### ".env file not found" in production

**Cause**: `.env` not deployed or wrong working directory

**Solution**:
1. Ensure `.env` exists in project root (same level as `quantagent/`)
2. Check working directory: `pwd` should be project root when starting app
3. Alternative: Export variables directly: `export DATABASE_URL=postgresql://...`

## Migration from Legacy Configuration

### Before (Legacy)

```python
from quantagent.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["api_key"] = "sk-..."
tg = TradingGraph(config=config)
```

### After (Current)

```bash
# .env file
OPENAI_API_KEY=sk-...
```

```python
from quantagent.trading_graph import TradingGraph

# Automatically reads from .env
tg = TradingGraph()
```

**Benefits**:
- No hardcoded credentials in Python files
- Configuration persists across restarts
- Web interface can update configuration
- Same code works across all environments

## References

**Configuration files**:
- `.env.example` - Configuration template
- `quantagent/settings.py` - Configuration module
- `quantagent/default_settings.py` - Model defaults

**Usage examples**:
- `quantagent/trading_graph.py` - LLM and database configuration
- `quantagent/database.py` - Database configuration
- `apps/flask/web_interface.py` - Runtime configuration updates
- `alembic/env.py` - Migration configuration

**Documentation**:
- `docs/MIGRATIONS.md` - Database setup guide
- `README.md` - Quick start guide
