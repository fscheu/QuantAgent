# Datos locales (no versionados)

Esta carpeta contiene archivos y datos locales necesarios para el desarrollo, backtesting y validación de M2 que no se publican en el repositorio git por términos de licencia o tamaño.

**Regla:** Todo dato local nuevo va en esta carpeta (`local-data/`) y se anota en este README en la misma entrega. Ningún dato local se comitea al repositorio git (todos los archivos excepto este `README.md` están ignorados en `.gitignore`).

**Ubicación:** Esta carpeta vive en la raíz del repositorio (`local-data/`) y no bajo `data/local/` porque `data/` pertenece a root en la VM (montada por el contenedor `quantagent_qa`).

---

## Inventario de archivos locales

### 1. `etf-1d-2026-10/`
- **Qué es:** Snapshot Parquet de 12 ETFs diarios (2007-01-03 a 2026-10-09) con su `manifest.json`.
- **Quién lo usa:** `backtest run --snapshot etf-1d-2026-10`, `scripts/crossval_compare.py`, `scripts/bt_adapter.py`, oráculos y tests de regresión.
- **Comando de verificación:**
  ```bash
  python -m quantagent.cli data snapshot verify --name etf-1d-2026-10
  ```
- **¿Se puede regenerar igual?:** No. Yahoo recalcula los precios históricos en cada dividendo o split nuevo; recrearlo daría hashes sha256 distintos e invalidaría el manifiesto versionado en `tests/fixtures/snapshots/etf-1d-2026-10.json`. Conservar en solo lectura (`chmod -R 555`).

### 2. `golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100.csv`
- **Qué es:** Resultado golden de referencia de trades para RSI sobre SPY diario 2007-2018 con `--costos etf` y 100% de capital (QuantAgent-688, PR #112).
- **Quién lo usa:** `tests/test_golden_snapshot.py`.
- **Comando de verificación / generación:**
  - Verificación:
    ```bash
    python -m pytest -q -m snapshot tests/test_golden_snapshot.py
    ```
  - Generación (si hiciera falta recrear):
    ```bash
    python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 --out local-data/golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100.csv
    ```
- **¿Se puede regenerar igual?:** Sí, siempre que se cuente con el snapshot `etf-1d-2026-10` y el motor conserve su comportamiento. Solo lectura.

### 3. `reserva-accesos.log`
- **Qué es:** Registro de aperturas de la reserva out-of-sample que `backtest run --abrir-reserva` escribe en esta carpeta (`quantagent/cli/backtest.py`). Cada línea registra fecha/hora UTC y el comando ejecutado al acceder a datos a partir de `QUANTAGENT_RESERVA_DESDE` (default 2023-01-01).
- **Quién lo usa:** Auditoría de accesos a la reserva en backtests y revisiones del PM.
- **¿Se puede regenerar igual?:** No se borra ni se regenera. Es el historial acumulativo de accesos a la reserva.

---

## Configuración de entorno

La variable de entorno `QUANTAGENT_SNAPSHOT_DIR` apunta a esta carpeta. `loop/config.env` (`LOOP_SNAPSHOT_DIR`), el valor por defecto de `scripts/loop/run_nightly.sh` y `scripts/loop/try.sh` apuntan a la ruta absoluta de `local-data/` del checkout principal, para que el loop y los worktrees puedan acceder a estos datos sin duplicar archivos.

## Factores mensuales hasta 2018-12 (QuantAgent-e4jc.7)

Receta de investigación interna: BAB de [AQR](https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly),
FF5, momentum y 49 industrias de [Kenneth French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
Sirven para elegir familias y atribuir retornos; no son precios operables ni se conectan a estrategias.
Este primer inventario no incluye QMJ, carry ni momentum de series de tiempo.
Respeta la decisión 14 del [DC R01](../docs/04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md).

Instalá la dependencia de esta receta con `python -m pip install openpyxl==3.1.5`.
Desde la raíz, ejecutá el bloque con este comando (sin copiar el código a otro archivo):

```bash
python -c 'from pathlib import Path; exec(Path("local-data/README.md").read_text().split("```python\n")[1].split("```")[0])'
```

Los originales se descargan en memoria y no se guardan. Solo se escriben CSV cortados
en `local-data/factores-2018/`, junto con `manifest.json` (URL, hash del original y del CSV,
unidad, filas y extremos). Todo queda ignorado por git, incluidos los derivados AQR.
Las descargas actuales pueden revisar historia anterior: los hashes identifican esa versión,
no un snapshot conocido en 2018. No se muestran retornos de 2019–2022 ni de la reserva.
La receta verifica el corte, los hashes de cada CSV y la exclusión de git antes de terminar.

```python
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import openpyxl

out = Path("local-data/factores-2018")
out.mkdir(parents=True, exist_ok=True)
manifest = {}
aqr = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
french = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
sources = {"aqr-bab": aqr + "Betting-Against-Beta-Equity-Factors-Monthly.xlsx"}
for name in ("F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor", "49_Industry_Portfolios"):
    sources[name] = french + name + "_CSV.zip"
for name, url in sources.items():
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    if name == "aqr-bab":
        sheet = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)["BAB Factors"]
        rows = iter(sheet.values)
        header = next(row for row in rows if row[0] == "DATE")
        dated = [(datetime.strptime(row[0], "%m/%d/%Y").strftime("%Y%m"), row[1:30])
                 for row in rows if isinstance(row[0], str) and re.fullmatch(r"\d{2}/\d{2}/\d{4}", row[0])]
        columns, unit = list(header[1:30]), "decimal"
    else:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        lines = archive.read(archive.namelist()[0]).decode("utf-8-sig").splitlines()
        start = next(i for i, line in enumerate(lines) if re.match(r"^\s*\d{6},", line))
        columns, unit = next(csv.reader([lines[start - 1]]))[1:], "percent"
        dated = []
        for row in csv.reader(lines[start:]):
            if not row or not re.fullmatch(r"\d{6}", row[0].strip()):
                break  # Solo el primer bloque mensual; no anual ni ponderación alternativa.
            dated.append((row[0].strip(), row[1:]))
    kept = [(date, values) for date, values in dated if date <= "201812"]
    assert kept and max(date for date, _ in kept) <= "201812"
    path = out / (name + ".csv")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["month", *columns])
        writer.writerows((date, *values) for date, values in kept)
    manifest[name] = dict(url=url, source_sha256=hashlib.sha256(raw).hexdigest(),
                          csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), unit=unit,
                          rows=len(kept), first=kept[0][0], last=kept[-1][0])
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for name, entry in json.loads((out / "manifest.json").read_text()).items():
    path = out / (name + ".csv")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["csv_sha256"]
    with path.open() as stream:
        assert all(row["month"] <= "201812" for row in csv.DictReader(stream))
    subprocess.run(["git", "check-ignore", "-q", str(path)], check=True)
    print(f"{name}: {entry['rows']} filas, {entry['first']}–{entry['last']}; hash e ignore OK")
```

French conserva sus unidades porcentuales y sus sentinelas de faltantes (-99.99/-999);
AQR conserva decimales y celdas vacías. No convertir faltantes en retornos al atribuir.
