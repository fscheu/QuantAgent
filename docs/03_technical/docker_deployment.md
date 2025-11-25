# Docker Deployment: Lightweight Setup

## Philosophy

**Don't overcomplicate**: Use Docker for database only during development, deploy full stack later.

```
Development:  Docker (PostgreSQL) + Local Python (your IDE)
Production:   Docker (all services)
```

---

## Development Setup

### Dockerfile

**File**: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy code
COPY . .

# Default: run tests
CMD ["pytest", "tests/"]
```

### docker-compose.yml

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  # PostgreSQL for all environments
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: quantagent_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Optional: Redis for caching (Phase 2)
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"

volumes:
  postgres_data:
```

---

## Development Workflow

### First Time Setup

```bash
# 1. Clone repo
git clone <repo>
cd quantagent

# 2. Start PostgreSQL in Docker
docker-compose up -d db

# 3. Create Python venv (local)
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# 4. Install dependencies (local)
pip install -r requirements-dev.txt

# 5. Run migrations (local)
alembic upgrade head

# 6. Run tests (local, fast)
pytest tests/
```

### Daily Development

```bash
# PostgreSQL already running from yesterday
# Just activate venv and code

source venv/bin/activate
pytest tests/  # Fast, local
python quantagent/dashboard/streamlit_app.py  # Run locally
```

### If PostgreSQL Stops

```bash
# Restart database
docker-compose up -d db

# Or check status
docker-compose ps
```

---

## Why This Works

### Development Benefits
- ✅ **Fast iterations**: Python code runs locally (instant feedback)
- ✅ **IDE integration**: Use your IDE (PyCharm, VS Code, etc)
- ✅ **Debugging**: Easy to debug with breakpoints
- ✅ **No overhead**: Not containerizing your code during dev

### PostgreSQL in Container Benefits
- ✅ **No installation needed**: No `brew install postgresql`
- ✅ **Clean**: Isolated from system
- ✅ **Portable**: Same image everywhere
- ✅ **Easy to wipe**: `docker-compose down -v` resets DB

---

## Production Deployment

### Multi-Service docker-compose.yml

**File**: `docker-compose.prod.yml` (Phase 2)

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: quantagent_prod
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  api:
    build: .
    command: uvicorn quantagent.api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db/quantagent_prod
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  scheduler:
    build: .
    command: python -m quantagent.scheduler.main
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db/quantagent_prod
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - db
    restart: unless-stopped

  ui:
    image: node:18-alpine
    working_dir: /app
    volumes:
      - ./ui:/app
    command: npm run build && npm run start
    ports:
      - "80:3000"
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
```

**Deploy**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Testing in Docker

### Run Tests in Container

```bash
# Build image
docker build -t quantagent .

# Run tests
docker run -e DATABASE_URL=postgresql://postgres:password@db/quantagent_dev \
  quantagent pytest tests/
```

### Or Use Docker Compose

```bash
# Run tests against database service
docker-compose run --rm app pytest tests/
```

---

## Environment Variables

**File**: `.env` (add to .gitignore)

```
DATABASE_URL=postgresql://postgres:password@localhost/quantagent_dev
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Usage**:
```python
import os

db_url = os.getenv("DATABASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
```

---

## Troubleshooting

### PostgreSQL won't start

```bash
# Check logs
docker-compose logs db

# Rebuild
docker-compose down -v
docker-compose up -d db
```

### Can't connect to database from local Python

```bash
# Check if db is running
docker-compose ps

# Check port is exposed
docker port quantagent-db-1 5432

# Try connecting directly
psql -h localhost -U postgres -d quantagent_dev
```

### Port already in use

```bash
# Change port in docker-compose.yml
# services.db.ports: ["5433:5432"]  # Use 5433 instead

# Or kill existing process
lsof -i :5432
kill -9 <PID>
```

---

## Benefits Summary

| Aspect | Benefit |
|--------|---------|
| **Portability** | Anyone can run `docker-compose up -d db` |
| **Consistency** | Same database everywhere |
| **Simplicity** | No manual PostgreSQL installation |
| **Cleanup** | Easy to reset: `docker-compose down -v` |
| **Corporate** | Works without admin privileges |
| **Production** | Same container image in production |

---

## Next Steps

1. **Week 1**: Setup Dockerfile + docker-compose.yml
2. **Local dev**: Run `docker-compose up -d db`, code locally
3. **Team**: They run same command, get same DB
4. **Phase 2**: Add API + scheduler services to docker-compose.prod.yml

