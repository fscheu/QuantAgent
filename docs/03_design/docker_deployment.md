# Docker Deployment: Lightweight Setup

## Philosophy

**Don't overcomplicate**: Use Docker for database only during development, deploy full stack later.

```
Development:  Docker (PostgreSQL) + Local Python (your IDE)
Production:   Docker (all services: DB + Flask app)
```

---

## Development Setup

### Dockerfile

**File**: `Dockerfile`

Lightweight container for **testing and CI/CD only**. Does NOT run the main app in dev (you run it locally in your IDE).

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal for TA-Lib + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default: run tests
CMD ["pytest", "tests/"]
```

**Why lightweight?**
- No `requirements-dev.txt` in the base image (keeps image small)
- Only production dependencies included
- Designed for test execution, not daily development
- Dev dependencies installed locally in your IDE

### docker-compose.yml

**File**: `docker-compose.yml`

PostgreSQL database service for development and testing.

```yaml
version: '3.8'

services:
  # PostgreSQL Database (for both development and testing)
  db:
    image: postgres:15
    container_name: quantagent-db
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
    networks:
      - quantagent-network

  # Optional: Redis for caching (Phase 2)
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"

volumes:
  postgres_data:

networks:
  quantagent-network:
    driver: bridge
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

# 5. Run tests (local, fast)
pytest tests/

# 6. Run Flask app locally (connects to Docker PostgreSQL)
python apps/flask/web_interface.py
# Access at http://127.0.0.1:5000
```

### Daily Development

```bash
# PostgreSQL already running from yesterday
# Just activate venv and code

source venv/bin/activate
pytest tests/  # Fast, local

# Run Flask app (connects to Docker DB on localhost:5432)
python apps/flask/web_interface.py
```

### Connecting to Docker PostgreSQL from Local Python

When you run Python code locally, PostgreSQL is inside Docker. The connection is:

```
localhost:5432 → Docker network → quantagent-db container
```

**Connection string** (for databases or env vars):
```
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantagent_dev
```

**Test the connection**:
```bash
# Install psql client if needed
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql-client

psql -h localhost -U postgres -d quantagent_dev
# Password: password
```

### If PostgreSQL Stops

```bash
# Restart database
docker-compose up -d db

# Or check status
docker-compose ps

# View database logs
docker-compose logs db
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

### Multi-Service docker-compose.prod.yml

**File**: `docker-compose.prod.yml` (Phase 2: Full Stack in Docker)

For production, containerize everything: PostgreSQL + Flask app.

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    container_name: quantagent-prod-db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: quantagent_prod
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - quantagent-prod

  app:
    build: .
    container_name: quantagent-app
    command: python apps/flask/web_interface.py
    ports:
      - "5000:5000"
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db:5432/quantagent_prod
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      FLASK_ENV: production
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - quantagent-prod

volumes:
  postgres_data:

networks:
  quantagent-prod:
    driver: bridge
```

**Deploy**:
```bash
# Create .env.prod with production credentials
echo "DB_PASSWORD=your_secure_password" > .env.prod
echo "OPENAI_API_KEY=sk-..." >> .env.prod
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env.prod

# Start full stack
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d
```

**Monitor**:
```bash
# View logs
docker-compose -f docker-compose.prod.yml logs app
docker-compose -f docker-compose.prod.yml logs db

# Access app at http://localhost:5000
```

---

## Testing in Docker

### Option 1: Run Tests Locally (Recommended for Dev)

```bash
# Fastest feedback loop during development
source venv/bin/activate
pytest tests/
```

### Option 2: Run Tests in Container (CI/CD)

```bash
# Build image
docker build -t quantagent .

# Run tests with PostgreSQL running
docker-compose up -d db
docker run -e DATABASE_URL=postgresql://postgres:password@localhost:5432/quantagent_dev \
  --network host \
  quantagent pytest tests/
```

### Option 3: Multi-Container Testing

```bash
# Start database service
docker-compose up -d db

# Run test container connected to the same network
docker run -e DATABASE_URL=postgresql://postgres:password@quantagent-db:5432/quantagent_dev \
  --network quantagent-network \
  quantagent pytest tests/
```

### Cleanup After Tests

```bash
# Stop database (keeps data)
docker-compose stop db

# Reset database (deletes all data)
docker-compose down -v db
```

---

## Environment Variables

### Development (.env)

**File**: `.env` (add to `.gitignore` - do NOT commit!)

```bash
# Database (PostgreSQL in Docker)
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantagent_dev

# LLM API Keys (required for agents)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Flask settings
FLASK_ENV=development
DEBUG=True
```

### Production (.env.prod)

```bash
# Database (PostgreSQL in Docker, different password)
DATABASE_URL=postgresql://postgres:your_secure_password@db:5432/quantagent_prod

# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Flask settings
FLASK_ENV=production
DEBUG=False
```

**Usage in Python**:
```python
import os

db_url = os.getenv("DATABASE_URL")
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
```

**Important**: Never commit `.env` or `.env.prod` files!

---

## Troubleshooting

### PostgreSQL won't start

```bash
# Check logs
docker-compose logs db

# Rebuild database service
docker-compose down -v db
docker-compose up -d db
```

### Can't connect to database from local Python

```bash
# Check if container is running
docker-compose ps

# Verify port is listening
docker-compose port db 5432
# Output should be: 0.0.0.0:5432

# Test connection with psql
psql -h localhost -U postgres -d quantagent_dev
# Password: password
```

### "Connection refused" when running Flask app locally

```bash
# Ensure Docker container is running
docker-compose up -d db

# Check Docker is accessible on localhost
docker-compose logs db | grep "ready"
# Should see: "database system is ready to accept connections"

# Verify DATABASE_URL in your .env is correct
# Should be: postgresql://postgres:password@localhost:5432/quantagent_dev
# NOT: postgresql://postgres:password@db:5432/quantagent_dev (that's for containers)
```

### Port 5432 already in use

```bash
# Option 1: Change port in docker-compose.yml
# Edit services.db.ports: ["5433:5432"]
# Then update DATABASE_URL to localhost:5433

# Option 2: Kill existing process
lsof -i :5432
kill -9 <PID>

# Option 3: Use Docker to find what's using the port
docker ps -a
```

### "Database does not exist" when connecting

```bash
# Verify database was created
docker-compose exec db psql -U postgres -l

# If quantagent_dev doesn't exist, restart with fresh volume
docker-compose down -v db
docker-compose up -d db
```

### Tests fail with "Connection refused"

```bash
# Ensure database is running
docker-compose up -d db

# Wait for health check to pass
docker-compose logs db | tail -5

# Run tests with proper DATABASE_URL
export DATABASE_URL=postgresql://postgres:password@localhost:5432/quantagent_dev
pytest tests/
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

### Phase 1: Development (Current)
- ✅ `docker-compose up -d db` for local development
- ✅ Flask app runs on local Python with IDE
- ✅ All team members use same PostgreSQL setup
- ✅ Easy to reset: `docker-compose down -v`

### Phase 2: Production (Future)
- Create `docker-compose.prod.yml` with full stack
- Containerize Flask app in addition to database
- Set up proper environment secrets management
- Test deployment workflow in staging

### Phase 3: Advanced (Optional)
- Add Redis caching service
- Implement database backups
- Add monitoring/logging infrastructure
- Multi-node cluster support

