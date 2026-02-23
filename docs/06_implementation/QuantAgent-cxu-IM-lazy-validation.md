# QuantAgent-cxu — Lazy validation en settings/database

**Issue:** QuantAgent-cxu
**Fecha:** 2026-02-23
**Estado:** Done

## Problema

Los módulos `settings.py` y `database.py` lanzaban `ValueError` al import time si faltaban variables de configuración (`DATABASE_URL`, `OPENAI_API_KEY`, etc.). Esto impedía que los unit tests en CI corrieran sin API keys reales ni base de datos configurada.

## Cambios realizados

### `quantagent/settings.py`
- Nueva función `require(name: str) -> str` para validación lazy.
- Las variables de módulo siguen leyendo de env con default vacío (nunca explotan).
- `require()` lanza `ValueError` solo cuando se invoca — es decir, cuando el valor realmente se necesita en runtime.

### `quantagent/database.py`
- `Base` se puede importar sin `DATABASE_URL` configurada.
- Engine y session factory se inicializan lazy (primera vez que se usan).
- Proxies `_LazyEngine` y `_LazySessionLocal` mantienen compatibilidad con código que accede a `database.engine` y `database.SessionLocal` directamente.
- `get_db()`, `init_db()`, `drop_all_tables()` usan `_get_engine()` interno.

### CI (`.github/workflows/main-ci.yml`)
- Se agregó service container PostgreSQL 16 para tests.
- `DATABASE_URL` se pasa como env var al step de tests.
- No se necesitan API keys dummy: los unit tests importan módulos sin explotar.

## Patrón

```
Leer config al import  →  siempre OK (valor puede ser "")
Usar config en runtime →  settings.require("VAR") → ValueError si vacío
```

## Verificación

```bash
# Import sin keys → OK
DATABASE_URL="" OPENAI_API_KEY="" python3 -c "from quantagent import database; print(database.Base)"

# Uso sin keys → ValueError descriptivo
DATABASE_URL="" python3 -c "from quantagent.database import get_db; next(get_db())"
```

## Archivos tocados
- `quantagent/settings.py`
- `quantagent/database.py`
- `.github/workflows/main-ci.yml`
- `pyproject.toml` (config ruff)
