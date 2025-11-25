# Workflow Integration: Dev Tools Setup

Cómo integrar las herramientas de desarrollo en tu flujo de trabajo diario para máxima eficiencia.

---

## 1. VSCode On-Save (Recomendado para desarrollo local)

Configura VSCode para ejecutar formateadores y linters automáticamente al guardar.

### Instalación de extensiones necesarias

En VSCode, instala:
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Black Formatter** (Microsoft)

### Configuración en VSCode

Abre `.vscode/settings.json` (crea si no existe):

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },

  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,

  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],

  "isort.args": ["--profile", "black", "--line-length", "100"],

  "python.linting.pylintPath": "pylint",
  "python.linting.flake8Path": "flake8",
  "python.linting.flake8Args": ["--max-line-length=100"]
}
```

### ¿Qué hace?

✅ **Al guardar un archivo:**
1. **Black** lo formatea automáticamente
2. **isort** reorganiza los imports
3. **Flake8** y **Pylint** muestran warnings en el editor (subrayados rojos/amarillos)
4. **mypy** valida tipos

**Ventaja:** Feedback instantáneo mientras escribes.

---

## 2. Pre-commit Hooks (Bloquea commits problemáticos)

Los hooks se ejecutan automáticamente ANTES de hacer `git commit`. Si algo falla, bloquea el commit.

### Instalación

```bash
pip install pre-commit
```

### Configuración

Crea archivo `.pre-commit-config.yaml` en la raíz del repo:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile=black", "--line-length=100"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: ["types-requests"]
        args: ["--ignore-missing-imports"]
```

### Activación

```bash
# Primera vez: instala los hooks
pre-commit install

# Prueba manual (sin hacer commit)
pre-commit run --all-files

# Después, cada commit ejecutará automáticamente los hooks
git commit -m "Tu mensaje"
```

### ¿Qué pasa?

✅ **Al hacer `git commit`:**
1. Se ejecutan Black, isort, flake8, mypy
2. Si hay problemas:
   - **Black/isort:** Modifica los archivos automáticamente → reintentas commit
   - **flake8/mypy:** Muestra errores → debes corregir manualmente
3. Solo si todo pasa, se completa el commit

**Ventaja:** No dejas código mal formateado en el repositorio.

---

## 2.5. Testing Workflow (Local)

Antes de hacer commit, ejecuta tests localmente para asegurar que la funcionalidad no se rompe.

### Ejecución Local de Tests

```bash
# Ejecuta todos los tests
pytest

# Tests con output detallado
pytest -v

# Solo tests rápidos (excluye 'slow')
pytest -m "not slow"

# Reporte de cobertura
pytest --cov=quantagent

# Test específico
pytest tests/test_trading_graph.py::TestIndicatorAgent::test_rsi_computation
```

### ¿Qué Testing Workflow?

```
1. Escribe código nuevo
   ↓
2. Escribe/actualiza tests en tests/
   ↓
3. Ejecuta: pytest -v
   ↓
4. Si ✅ tests pasan:
   → Haz commit
   Sino → Arregla código
```

### Fixtures de conftest.py en Tests

El archivo `tests/conftest.py` proporciona fixtures reutilizables:

- **Datos:** `sample_ohlcv_data`, `sample_state` - datos simulados sin API calls
- **Mocks:** `patch_yfinance`, `patch_talib` - evita llamadas reales a APIs
- **Config:** `mock_config`, `mock_env_vars` - configuración de test
- **Directorios:** `temp_output_dir`, `temp_chart_dir` - aislamiento de archivos

**Ejemplo de test con fixture:**

```python
def test_rsi_calculation(sample_ohlcv_data):
    """Test RSI computation."""
    from graph_util import TechnicalTools

    tools = TechnicalTools()
    rsi = tools.compute_rsi(sample_ohlcv_data["close"])
    assert len(rsi) == 30
    assert all(0 <= val <= 100 for val in rsi)
```

### Marcadores de Tests

Clasifica tests por tipo para ejecutar selectivamente:

```bash
# Solo tests de integración
pytest -m integration

# Excluir tests que requieren APIs
pytest -m "not api"

# Solo tests de vision LLMs
pytest -m vision

# Combinaciones
pytest -m "api and not slow"
```

**Tipos de marcadores disponibles:**
- `@pytest.mark.slow` - tests que toman tiempo
- `@pytest.mark.integration` - tests de múltiples componentes
- `@pytest.mark.api` - tests que usan APIs externas
- `@pytest.mark.vision` - tests de vision LLMs

---

## 3. GitHub Actions / CI-CD (Verificación remota)

En el repositorio remoto (GitHub), verifica automáticamente que todos los cambios cumplan con los estándares, incluyendo tests.

### Configuración

Crea `.github/workflows/quality-checks.yml`:

```yaml
name: Code Quality Checks

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11']

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Black check
        run: black --check .

      - name: isort check
        run: isort --check-only .

      - name: Flake8
        run: flake8 .

      - name: Pylint
        run: pylint --recursive=y . || true  # || true para no fallar en primer error

      - name: mypy
        run: mypy . || true  # || true para warning, no error

      - name: pytest
        run: pytest --cov=quantagent --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: false
```

### ¿Qué pasa?

✅ **En cada `git push` o PR:**
1. GitHub ejecuta automáticamente todas las herramientas
2. Si algo falla, la PR muestra ❌ (rojo)
3. Si todo pasa, muestra ✅ (verde)
4. No puedes mergear si hay checks fallidos

**Ventaja:** Garantiza código limpio en main, sin excepciones.

---

## Recomendación de Flujo Completo

### Para el developer (local)

```
1. Escribe código + tests
   ↓
2. VSCode guarda → Black/isort/flake8 automático
   ↓
3. Ejecuta: pytest -v
   ↓
4. Pre-commit hooks verifican (Black/isort/flake8/mypy)
   ↓
5. git commit -m "mensaje"
```

### Para el repositorio (remoto)

```
6. git push origin rama
   ↓
7. GitHub Actions ejecuta:
   - Black check
   - isort check
   - Flake8
   - Pylint
   - mypy
   - pytest (con coverage)
   ↓
8. Si ✅ todas pasan → puedes mergear PR
   Si ❌ alguna falla → revisar logs y hacer push nuevamente
   ↓
9. Coverage report se sube a Codecov
```

---

## Setup Rápido (Copiar y Pegar)

### 1. Instala dependencias
```bash
pip install -r requirements-dev.txt
pip install pre-commit
```

### 2. Configura VSCode
Copia el contenido de la sección anterior al `.vscode/settings.json`

### 3. Configura pre-commit
```bash
# Copia el .pre-commit-config.yaml de arriba
pre-commit install
```

### 4. Configura GitHub Actions
```bash
# Crea directorio
mkdir -p .github/workflows

# Copia el quality-checks.yml de arriba
```

### 5. Prueba tests locales
```bash
# Ejecuta todos los tests
pytest -v

# Con cobertura
pytest --cov=quantagent
```

### 6. Prueba todo
```bash
# Test pre-commit
pre-commit run --all-files

# Intenta hacer commit
git add .
git commit -m "Setup dev tools"
```

---

## Archivos a Crear/Editar

| Archivo | Propósito | Ubicación |
|---------|-----------|-----------|
| `.vscode/settings.json` | Configuración VSCode | Raíz del repo |
| `.pre-commit-config.yaml` | Hooks pre-commit | Raíz del repo |
| `.flake8` | Config flake8 | Raíz del repo |
| `pytest.ini` | Config pytest | Raíz del repo |
| `pyproject.toml` | Config black/isort/mypy/pytest | Raíz del repo |
| `tests/conftest.py` | Fixtures y config de tests | Dentro de `tests/` |
| `.github/workflows/quality-checks.yml` | CI/CD GitHub (incl. pytest) | Dentro de `.github/workflows/` |

---

## Troubleshooting

### VSCode no ejecuta formateo al guardar
- Verifica que Black esté instalado: `pip show black`
- Recarga VSCode: Ctrl+Shift+P → "Reload Window"
- Asegúrate que Python sea el intérprete por defecto

### Pre-commit hook falla
```bash
# Ver qué falla
pre-commit run --all-files --verbose

# Saltarse un hook (solo en emergencias)
git commit --no-verify -m "Mensaje"  # ⚠️ No recomendado
```

### GitHub Actions toma mucho tiempo
- Aumenta timeout en `.yml`
- Considera hacer checks menos rigurosos en CI (remoto) vs local

---

## Best Practices

✅ **DO:**
- Escribe tests cuando agregas funcionalidad nueva
- Ejecuta `pytest -v` antes de hacer commit
- Ejecuta `pre-commit run --all-files` antes de abrir PR
- Mantén `.vscode/settings.json` en el repo para que todo el equipo use la misma config
- Usa `--no-verify` solo en casos de emergencia (hot fixes)
- Revisa logs de GitHub Actions si tu PR falla

❌ **DON'T:**
- Deshabilita linters para "pasar rápido"
- Ignores warnings de mypy sin revisar
- Hagas commits sin ejecutar checks locales
- Hagas commits sin pasar tests locales
- Mergees PRs que fallen en CI/CD
- Ignores reportes de cobertura (coverage) en Codecov

---

## Configuración Completa (Archivos de Ejemplo)

### `.vscode/settings.json`
```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "isort.args": ["--profile", "black", "--line-length", "100"]
}
```

### `pyproject.toml`
```toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
warn_return_any = true
disallow_untyped_defs = false
```

### `.flake8`
```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .git,__pycache__,venv,.pytest_cache
```

---

## Próximos Pasos

1. **Hoy:** Configura VSCode settings.json
2. **Mañana:** Instala pre-commit y crea `.pre-commit-config.yaml`
3. **Esta semana:** Configura GitHub Actions para CI/CD
4. **Opcional:** Agrega configuraciones adicionales en `pyproject.toml`

Así tendrás un flujo de trabajo profesional que garantiza código limpio en cada paso.
