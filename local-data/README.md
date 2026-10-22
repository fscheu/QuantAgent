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

## Regla de admisión de ETFs ampliados (QuantAgent-e4jc.8.1)

Corte histórico: **2007-12-31, cierre de mercado**. Esta regla se escribe ahora;
no se afirma que existiera en 2007. Se integra en un commit propio; después D-2b
documenta los empalmes enlazando ese commit y recién D-2c investiga candidatos.
No contiene símbolos ni autoriza descargas, empalmes o backtests.

Un instrumento se admite si cumple **todas** estas condiciones, según documentos
del emisor o registros de bolsa/regulador referidos a su situación al corte:

1. Cotizaba en una bolsa estadounidense, en USD, con lanzamiento y primera
   sesión negociable no posteriores al corte; seguía activo en esa fecha.
   Fecha de creación del índice o historia simulada no cuentan como lanzamiento.
2. Era un fondo cotizado o trust cotizado con exposición a una de estas clases:
   moneda extranjera frente al USD; crédito corporativo o soberano emergente;
   bonos soberanos internacionales; renta variable de un país; o renta variable
   mediante un inverso diario 1x. La clase surge del mandato vigente al corte.
3. Su exposición objetivo era larga sin apalancamiento (1x), salvo los inversos
   de renta variable con objetivo diario -1x, admitidos explícitamente.
   Se excluyen múltiplos distintos de +1x/-1x y los ETN (deuda del emisor).
4. No tenía clasificación fiscal de publicly traded partnership (PTP).
   El uso de derivados por sí solo no determina esa clasificación; se requiere
   evidencia de la estructura fiscal vigente al corte.

La evaluación posterior registrará por instrumento clase, estructura, objetivo,
fechas de lanzamiento/primera sesión y baja si existe, fuente (URL y fecha del
documento), decisión sí/no por condición y motivo. Un dato sin evidencia queda
como no verificable y no se admite; no se presume que cumple. Una fuente publicada
después puede acreditar hechos históricos, pero no justificar con información
posterior una condición que no se cumplía al corte.

No se filtra por rentabilidad, Sharpe, volatilidad, correlación, volumen o AUM
posteriores, ni por seguir disponible hoy. Se buscarán también fondos liquidados,
fusionados o retirados después del corte en registros históricos, conservando su
elegibilidad y documentando la baja; no se reemplazan por ganadores supervivientes.
Los cerrados antes del corte y los lanzados después quedan excluidos.
Si faltan precios de un desaparecido elegible, se registra como dato pendiente:
no se lo borra del universo ni se presenta el subconjunto disponible como libre
de sesgo de supervivencia. Ningún fondo aporta velas operables previas a su primera
sesión; cualquier proxy requiere el documento separado D-2b, sin retornos futuros.

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

## Futuros de pysystemtrade hasta 2024-03 (QuantAgent-e4jc.9)

Inventario para investigación de tendencia y carry, sin conexión al motor ni a Alpaca.
Fuente: [pysystemtrade de Robert Carver](https://github.com/robcarver17/pysystemtrade/tree/007feb5da6cde5b812cdbc1463e73d016393bc63/data/futures),
revisión fija del 25/03/2024. Se conservan los CSV originales de `adjusted_prices_csv`
(precio continuo ajustado) y `multiple_prices_csv` (precio, carry, forward y contratos).
`CARRY` es el precio de otro contrato, no un retorno ni una señal de carry calculada.
Los faltantes quedan vacíos; no se rellenan. La frecuencia puede ser intradiaria.

El [README de origen](https://github.com/robcarver17/pysystemtrade/blob/007feb5da6cde5b812cdbc1463e73d016393bc63/README.md#licensing-and-legal-stuff)
declara GNU GPL v3 y ausencia de garantía. La receta guarda su `LICENSE` y `README.md`
sin modificar junto a los datos; no redistribuye CSV ni incorpora código de pysystemtrade.
Todo queda ignorado en `local-data/pysystemtrade-2024-03/` (aproximadamente 700 MB).
El manifiesto identifica URL, revisión, fecha de descarga, hashes, columnas, filas y extremos.
Esto incorpora datos hasta marzo de 2024, pero no autoriza evaluar estrategias en
2019–2022 ni abrir la reserva 2023+: solo se comprueba carga y metadatos, sin resultados.

Desde la raíz, descargá y verificá el inventario con este comando (requiere red):

```bash
python -c 'from pathlib import Path; exec(Path("local-data/README.md").read_text().split("```python\n")[2].split("```")[0])'
```

```python
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ref = "007feb5da6cde5b812cdbc1463e73d016393bc63"
repo = "https://api.github.com/repos/robcarver17/pysystemtrade"
base = f"https://raw.githubusercontent.com/robcarver17/pysystemtrade/{ref}/"
out = Path("local-data/pysystemtrade-2024-03")
with urlopen(f"{repo}/git/trees/{ref}?recursive=1", timeout=60) as response:
    tree = json.load(response)
assert not tree["truncated"]
folders = ("data/futures/adjusted_prices_csv/", "data/futures/multiple_prices_csv/")
paths = sorted(item["path"] for item in tree["tree"]
               if item["path"].startswith(folders) and item["path"].endswith(".csv"))
assert paths
manifest = dict(revision=ref, downloaded_at=datetime.now(timezone.utc).isoformat(),
                license="GPL-3.0", files={})
for source in ["LICENSE", "README.md", *paths]:
    with urlopen(base + source, timeout=60) as response:
        raw = response.read()
    relative = source.removeprefix("data/futures/")
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    entry = dict(url=base + source, sha256=hashlib.sha256(raw).hexdigest())
    if source.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(raw), index_col="DATETIME", parse_dates=True)
        assert not frame.empty and frame.index.notna().all()
        assert frame.index.max() < pd.Timestamp("2024-04-01")
        entry.update(rows=len(frame), first=str(frame.index.min()),
                     last=str(frame.index.max()), columns=list(frame.columns))
    manifest["files"][relative] = entry
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for relative, entry in json.loads((out / "manifest.json").read_text())["files"].items():
    path = out / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    subprocess.run(["git", "check-ignore", "-q", str(path)], check=True)
subprocess.run(["git", "check-ignore", "-q", str(out / "manifest.json")], check=True)
sample = pd.read_csv(out / "multiple_prices_csv/SP500.csv", index_col="DATETIME", parse_dates=True)
assert {"PRICE", "CARRY", "FORWARD", "PRICE_CONTRACT", "CARRY_CONTRACT", "FORWARD_CONTRACT"} == set(sample.columns)
assert sample[["PRICE", "CARRY", "FORWARD"]].notna().all(axis=1).any()
print(f"{len(paths)} CSV: fuente, fechas, hashes e ignore OK")
print(f"SP500 cargado: {len(sample)} filas; {sample.index.min()} a {sample.index.max()}")
```
