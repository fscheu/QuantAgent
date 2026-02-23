# Manejo de dependencias con pyproject.toml

## Flujo básico

### Comparación con requirements.txt

| Acción | requirements.txt | pyproject.toml |
|--------|------------------|----------------|
| Agregar dep | Editar txt | Editar `dependencies = [...]` |
| Instalar | `pip install -r requirements.txt` | `pip install -e .` |
| Instalar + dev | (otro archivo) | `pip install -e ".[dev]"` |
| Ver qué hay instalado | `pip freeze` | `pip freeze` (igual) |

### Agregar una dependencia

1. Editar `pyproject.toml`, agregar en la lista `dependencies`:

```toml
dependencies = [
    "flask",
    "yfinance",
    # ... las demás ...
    "requests",
    "redis",        # <-- nueva dependencia
]
```

2. Instalar:

```bash
pip install -e .
```

### Dependencias de desarrollo

Para instalar también pytest, black, ruff, etc.:

```bash
pip install -e ".[dev]"
```

Estas están definidas en `[project.optional-dependencies]` en pyproject.toml.

---

## Modo editable (`-e`)

### ¿Qué es?

El flag `-e` instala el proyecto en modo "editable". La diferencia:

- **Sin `-e`**: pip copia tu código a `site-packages/`. Los cambios locales no se reflejan hasta reinstalar.
- **Con `-e`**: pip crea un enlace simbólico. Python lee directamente desde tu carpeta de trabajo.

### Ejemplo

```python
# quantagent/utils.py
def saludar():
    return "Hola"
```

```python
# En cualquier terminal con el venv activado:
>>> from quantagent.utils import saludar
>>> saludar()
"Hola"
```

Editás el archivo:

```python
# quantagent/utils.py
def saludar():
    return "Hola mundo"  # cambiaste esto
```

**Con `-e`**: Reiniciás Python y ya tenés el cambio:

```python
>>> from quantagent.utils import saludar
>>> saludar()
"Hola mundo"  # refleja el cambio inmediatamente
```

**Sin `-e`**: Seguirías viendo "Hola" hasta hacer `pip install .` de nuevo.

### Verificar modo editable

```bash
pip show quantagent
```

Si dice `Editable project location: /path/to/QuantAgent`, está en modo editable.

### ¿Por qué usar siempre `-e` en desarrollo?

Sin modo editable tendrías que reinstalar después de cada cambio de código, lo cual es impráctico durante desarrollo.
