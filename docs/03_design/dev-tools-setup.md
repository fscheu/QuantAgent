# Development Tools Setup Guide

Guía rápida para configurar y usar las herramientas de desarrollo en QuantAgent.

---

## Code Quality & Formatting

### Black (Formateador de Código)

**¿Qué hace?**
Formatea automáticamente tu código Python según un estilo consistente y determinístico. No necesitas debatir sobre espacios vs tabs; Black decide por ti.

**Documentación:** https://black.readthedocs.io/

**Instalación:**
```bash
pip install black>=23.0
```

**Uso básico:**
```bash
# Formatea todos los archivos .py en el directorio
black .

# Formatea un archivo específico
black apps/flask/web_interface.py

# Modo "check" (no modifica, solo reporta)
black --check .
```

**Configuración básica** (opcional, en `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py311']
include = '\.pyi?$'
```

---

### isort (Organizador de Imports)

**¿Qué hace?**
Organiza automáticamente los `import` de tus archivos Python de forma consistente (orden: stdlib → third-party → local).

**Documentación:** https://pycqa.github.io/isort/

**Instalación:**
```bash
pip install isort>=5.12
```

**Uso básico:**
```bash
# Ordena imports en todos los archivos
isort .

# Verifica sin modificar
isort --check-only .

# Archivo específico
isort apps/flask/web_interface.py
```

**Configuración básica** (en `pyproject.toml`):
```toml
[tool.isort]
profile = "black"
line_length = 100
```

---

### flake8 (Linter)

**¿Qué hace?**
Revisa tu código y reporta problemas de estilo, errores lógicos y violaciones de convenciones (PEP 8).

**Documentación:** https://flake8.pycqa.org/

**Instalación:**
```bash
pip install flake8>=6.0
```

**Uso básico:**
```bash
# Revisa todos los archivos
flake8 .

# Archivo específico
flake8 apps/flask/web_interface.py

# Mostrar estadísticas
flake8 --statistics .
```

**Configuración básica** (en `.flake8`):
```ini
[flake8]
max-line-length = 100
exclude = .git,__pycache__,venv
ignore = E203,W503
```

---

### pylint (Análisis Estático Profundo)

**¿Qué hace?**
Análisis más exhaustivo que flake8: detecta bugs potenciales, problemas de diseño, complejidad de código y proporciona scores de calidad.

**Documentación:** https://pylint.pycqa.org/

**Instalación:**
```bash
pip install pylint>=2.17
```

**Uso básico:**
```bash
# Analiza un archivo
pylint apps/flask/web_interface.py

# Todos los archivos (verbose)
pylint --recursive=y .

# Genera reporte HTML
pylint --output-format=html --reports=yes . > pylint_report.html
```

**Configuración básica** (en `.pylintrc`):
```ini
[MESSAGES CONTROL]
disable=missing-docstring,too-many-arguments

[FORMAT]
max-line-length=100
```

---

## Testing Framework

### pytest (Test Runner)

**¿Qué hace?**
pytest es un framework de testing que automatiza la ejecución de pruebas unitarias, de integración y funcionales. Te permite verificar que tu código hace lo que se espera.

**Documentación:** https://docs.pytest.org/

**Instalación:**
```bash
pip install pytest>=7.4 pytest-cov>=4.1 pytest-mock>=3.11
```

**Estructura Básica:**
```python
# tests/test_example.py
def test_addition():
    """Test simple function."""
    assert 2 + 2 == 4

def test_with_fixture(sample_ohlcv_data):
    """Test using a fixture (data provided by conftest.py)."""
    assert len(sample_ohlcv_data["close"]) == 30
```

**Ejecución:**
```bash
# Ejecuta todos los tests
pytest

# Tests verbosos (detallado)
pytest -v

# Test específico
pytest tests/test_example.py::test_addition

# Solo tests rápidos (excluyendo 'slow')
pytest -m "not slow"

# Con reporte de cobertura
pytest --cov=quantagent --cov-report=html
```

---

### conftest.py (Configuración de Tests)

**¿Qué hace?**
`conftest.py` es un archivo especial de pytest que define "fixtures" (datos/configuraciones reutilizables) y hooks de configuración. Se ejecuta automáticamente antes de los tests.

**Ubicación:** `tests/conftest.py`

**¿Qué incluimos?**

#### 1. **Fixtures de Datos**
Proporcionan datos de prueba sin repetir código:

```python
@pytest.fixture
def sample_ohlcv_data() -> Dict[str, List[float]]:
    """Proporciona datos OHLCV (Open, High, Low, Close, Volume) simulados."""
    return {
        "open": [100.0] * 30,
        "high": [101.0] * 30,
        "low": [99.0] * 30,
        "close": [100.5] * 30,
        "volume": [1000000.0] * 30,
    }
```

**Uso en test:**
```python
def test_indicator(sample_ohlcv_data):
    # sample_ohlcv_data se inyecta automáticamente
    assert len(sample_ohlcv_data["close"]) == 30
```

#### 2. **Fixtures de Configuración**
Para LLM mocks, variables de entorno, etc.:

```python
@pytest.fixture
def mock_config() -> Dict:
    """Configuración mock para tests."""
    return {
        "agent_llm_provider": "openai",
        "api_key": "sk-test-key",
    }

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Variables de entorno para tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
```

#### 3. **Fixtures de Mocking**
Evitan llamadas reales a APIs:

```python
@pytest.fixture
def patch_yfinance():
    """Mock yfinance para no hacer llamadas reales."""
    with patch("yfinance.download") as mock:
        mock.return_value = pd.DataFrame({...})
        yield mock
```

#### 4. **Fixtures de Directorios**
Crea directorios temporales para outputs:

```python
@pytest.fixture
def temp_chart_dir(tmp_path) -> Path:
    """Directorio temporal para gráficos."""
    chart_dir = tmp_path / "charts"
    chart_dir.mkdir()
    return chart_dir
```

**Fixtures disponibles en QuantAgent:**
- `sample_ohlcv_data` - Datos OHLCV simulados (30 candelas)
- `sample_state` - Estado completo IndicatorAgentState
- `mock_config` - Configuración con keys fake
- `mock_openai_llm`, `mock_anthropic_llm`, `mock_vision_llm` - LLMs mockeados
- `temp_output_dir`, `temp_chart_dir` - Directorios temporales
- `patch_yfinance`, `patch_talib` - Mocks de APIs externas

**Configuración en `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --strict-markers --tb=short"
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
    "api: marks tests requiring API calls",
    "vision: marks vision-dependent tests",
]
```

---

## Type Checking

### mypy (Type Checker)

**¿Qué hace?**
Verifica que los tipos de datos en tu código sean consistentes (requiere type hints). Previene errores tipo `AttributeError` en tiempo de desarrollo.

**Documentación:** https://mypy.readthedocs.io/

**Instalación:**
```bash
pip install mypy>=1.5
```

**Uso básico:**
```bash
# Verifica todos los archivos
mypy .

# Archivo específico
mypy trading_graph.py

# Modo strict (más riguroso)
mypy --strict .
```

**Configuración básica** (en `mypy.ini`):
```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False  # Empieza con False, luego sube a True
```

**Ejemplo de uso en el código:**
```python
from typing import Dict, List

def compute_rsi(prices: List[float], period: int = 14) -> Dict[str, float]:
    """Compute RSI indicator."""
    # mypy verificará que prices es Lista y retorna Dict
    pass
```

---

### types-requests (Type Stubs)

**¿Qué hace?**
Proporciona información de tipos para la librería `requests` (usada en trading_graph.py). Necesario si usas mypy y haces peticiones HTTP.

**Documentación:** https://github.com/python/typeshed

**Instalación:**
```bash
pip install types-requests>=2.31
```

No requiere configuración; mypy lo detecta automáticamente.

---

## Documentation (Opcional)

### Sphinx + sphinx-rtd-theme

**¿Qué hace?**
Sphinx genera documentación técnica automáticamente desde docstrings y archivos Markdown/RST. `sphinx-rtd-theme` aplica un tema limpio tipo Read the Docs.

**Documentación:**
- Sphinx: https://www.sphinx-doc.org/
- RTD Theme: https://sphinx-rtd-theme.readthedocs.io/

**Instalación:**
```bash
pip install sphinx>=7.0 sphinx-rtd-theme>=1.3
```

**Setup inicial:**
```bash
# En directorio docs/
sphinx-quickstart --quiet --project QuantAgent --author "Your Name" --release 1.0

# Edita docs/conf.py para agregar:
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']
html_theme = 'sphinx_rtd_theme'
```

**Generar documentación:**
```bash
# En el directorio docs/
make html

# Abre en navegador
open _build/html/index.html
```

---

## Workflow Recomendado

### Pre-commit Hook (Automatizar)

Crea `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Ejecuta herramientas antes de cada commit

isort .
black .
flake8 .
mypy .

if [ $? -ne 0 ]; then
  echo "❌ Quality checks failed"
  exit 1
fi

echo "✅ All checks passed!"
```

Hazlo ejecutable:
```bash
chmod +x .git/hooks/pre-commit
```

---

### Quick Commands

```bash
# Instalar todas las herramientas de desarrollo
pip install -r requirements-dev.txt

# Formatear código
black . && isort .

# Revisar calidad
flake8 . && pylint --recursive=y .

# Type checking
mypy .

# Todo de una vez (en Linux/Mac)
black . && isort . && flake8 . && mypy . && echo "✅ All checks passed!"
```

---

## Próximos Pasos

1. **Ejecuta** `pip install -r requirements-dev.txt`
2. **Prueba** `black . && isort .` en tu código
3. **Revisa** `flake8 .` y `mypy .` para entender warnings
4. **Opcional:** Configura pre-commit hooks para automatizar

---

## Referencias Rápidas

| Herramienta | Comando | Propósito |
|-------------|---------|----------|
| Black | `black .` | Formatea código |
| isort | `isort .` | Ordena imports |
| flake8 | `flake8 .` | Linting básico |
| pylint | `pylint .` | Análisis profundo |
| mypy | `mypy .` | Type checking |
| Sphinx | `make html` | Genera documentación |
