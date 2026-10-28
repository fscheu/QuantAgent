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

## Empalme de proxies de investigación (QuantAgent-e4jc.8.2)

Parte de la regla D-2a integrada en el [commit 0c7a0330](https://github.com/fscheu/QuantAgent/commit/0c7a0330dc4244191cbffe01822280c44e4d2028).
Convención fijada antes de seleccionar candidatos: corte **2007-12-31** y comienzo
del tramo de investigación **2007-01-03**. No crea series ni velas operables anteriores
al lanzamiento; un proxy se identifica por separado del ETF y de su retorno total.

| Clase elegible | Fuente gratuita y transformación fijada | Límites / decisión de empalme |
| --- | --- | --- |
| Moneda extranjera frente al USD | FX diario FRED, fuente Fed H.10: DEXUSAL (AUD), DEXUSUK (GBP) en USD/moneda; DEXCAUS (CAD), DEXSZUS (CHF), DEXJPUS (JPY) en moneda/USD. Para otras monedas se exige la misma evidencia de orientación. | Solo proxy spot de la misma moneda y exposición larga 1x; no reproduce intereses del depósito, carry, comisiones ni distribuciones del fondo. |
| Crédito corporativo o soberano emergente | Tasas FRED H.15 y DGS* solo como referencias de tasas estadounidenses, sin enlace al ETF. | Sin proxy defendible con estas fuentes: faltan spreads, impagos, recuperos, cupones y duración de la cartera. Conservar faltantes. |
| Bonos soberanos internacionales | FX H.10 y Treasury sintético DGS* como referencias separadas, sin combinarlos para enlazar el ETF. | No reconstruyen curvas locales, pesos por país, duración ni cobertura cambiaria del mandato histórico. Conservar faltantes. |
| Renta variable de un país | Ningún proxy admitido a partir de FX o tasas. | Faltan índice histórico, dividendos, composición y tratamiento fiscal; conservar faltantes. |
| Inverso diario -1x de renta variable | Ningún proxy admitido a partir de FX o tasas. | Negar el retorno acumulado de un índice no reproduce el rebalanceo diario, financiación ni gastos; conservar faltantes. |

Fuentes y unidades consultadas el 2026-10-22: [H.10](https://www.federalreserve.gov/releases/h10/current/)
(cotizaciones de mediodía de Nueva York), [DEXUSUK](https://fred.stlouisfed.org/series/DEXUSUK),
[DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS), [H.15](https://www.federalreserve.gov/releases/h15/)
y [DGS10](https://fred.stlouisfed.org/series/DGS10) (rendimiento anual en porcentaje,
vencimiento constante; no precio ni retorno). Los códigos FX son series de monedas,
no una selección de ETFs. La elección depende del mandato al corte, nunca del ajuste posterior.

Para FX, normalizar a `S_t` en USD por unidad extranjera: usar la cotización directa
o su recíproca según la tabla; `r_proxy,t = S_t / S_{t-1} - 1`, decimal, sin sumar carry.
Para tasas, `y_t = DGS10_t / 100`; ni `y_t` ni su variación son retornos de un bono.
Como referencia sintética separada, fijar un Treasury par de 10 años que se renueva
cada día observado, nominal 1, cupón semestral `c/2`, con `c = y_{t-1}`.
Su precio aproximado con curva plana es
`B(y,h;c) = sum(k=1..20, (c/2 + 1[k=20]) / (1+y/2)^(k-2h))`;
`r_sint,t = B(y_t,dt;c) / B(y_{t-1},0;c) - 1`, con `dt` días calendario/365.
Incluye devengo y repricing aproximados para intervalos menores a medio año;
no es un índice Treasury real: ignora curva completa, costos y cambios de composición.
No se elige otro plazo DGS* por correlación futura ni se traslada esta duración a crédito
o bonos internacionales. Este sintético no habilita empalmes de esas clases.

Fecha del enlace FX `L`: primera sesión desde `max(lanzamiento, 2007-01-03)` con precio
ajustado del ETF `A_L` y spot disponibles, no posterior al corte; sin ella no hay enlace.
`A` es el cierre ajustado por splits y dividendos de Yahoo del snapshot D-2c, en USD.
En cada segmento continuo, `P_t = product(u=t0+1..t, 1+r_proxy,u)` con `P_t0=1`.
Serie de investigación: `I_t=P_t/P_L` para `t<L`; `I_L=1`; `I_t=A_t/A_L` para `t>L`.
El primer retorno del ETF es `A_{L+1}/A_L-1`: no se calcula un salto entre niveles
de distintas unidades. Registrar `L` y ambos niveles originales, sin ajustar por regresión.
La normalización en L cambia la escala, no usa retornos posteriores para elegir el proxy.

Validación descriptiva: pares diarios comunes desde `L` hasta **2007-12-31**,
sin extender a 2008 para completar muestra. Informar fechas, cantidad de pares,
correlación y media/desvío de `r_ETF-r_proxy`; con menos de 60 pares, declarar muestra
insuficiente. No calibrar pesos, carry o fecha del enlace ni seleccionar candidatos
con esas métricas; no certifican equivalencia de retorno total. El horario spot difiere
del cierre del ETF. No descargar ni evaluar 2019–2022 ni la reserva 2023+ para validar.
Conservar ausencias como NA, sin interpolar, rellenar ni saltar una sesión faltante
para fabricar un retorno diario; un hueco interrumpe el segmento acumulado y el enlace
solo cubre el segmento que contiene L. Registrar fuente, fecha de descarga, versión/hash,
calendario y ajustes del ETF: una descarga actual puede revisar la historia y no prueba
disponibilidad en tiempo real en 2007; para señales se necesita fecha de publicación/vintage.

## Selección y snapshot ampliado (QuantAgent-e4jc.8.3)

Selección del **2026-10-22**, posterior a la regla integrada en `0c7a0330`
y al empalme D-2b en `516c2a85` (ambos ancestros de `origin/lote/investigacion-edge`).
Inventario inicial por mandato, no censo exhaustivo ni selección por resultados.
Las fuentes siguientes se consultaron ese día; documentos posteriores acreditan
hechos históricos, no disponibilidad de precios en tiempo real al corte.

En la tabla, GT = grantor trust fiscal (no partnership); RIC = fondo bajo
Subchapter M (no PTP). «Sí» significa C1/C2/C3/C4 = sí: cotización estadounidense
en USD activa al cierre de 2007, clase admitida, objetivo permitido y no PTP.
«Primera observada» acredita negociación como máximo en esa fecha; no confundir
inception/NAV con primera transacción bursátil. No se inventan sesiones anteriores.

| Candidato | Clase y objetivo al corte | Lanzamiento / primera observada en el recorte | Estructura y evidencia histórica | Decisión / cobertura hasta 2018-12-31 |
| --- | --- | --- | --- | --- |
| FXA | AUD/USD +1x | 2006-06-26 / 2007-01-03 | GT; [10-K 2020, Item 1 y nota 2](https://www.sec.gov/Archives/edgar/data/1353614/000156459021009023/fxa-10k_20201231.htm) | Sí; 3020 filas |
| FXB | GBP/USD +1x | 2006-06-26 / 2007-01-03 | GT; [10-K 2008, Item 1 y notas](https://www.sec.gov/Archives/edgar/data/1353611/000119312509006139/d10k.htm) | Sí; 3020 filas |
| FXC | CAD/USD +1x | 2006-06-26 / 2007-01-03 | GT; [10-K 2009, Item 1 y notas](https://www.sec.gov/Archives/edgar/data/1353612/000119312510006518/d10k.htm) | Sí; 3020 filas |
| FXY | JPY/USD +1x | 2007-02-13 / 2007-02-13 | GT; [10-Q julio 2008, Trust Overview y notas](https://www.sec.gov/Archives/edgar/data/1353613/000089706908001411/cmw3720.htm) | Sí; 2992 filas |
| LQD | Crédito corporativo IG +1x | 2002-07-22 / 2007-01-03 | RIC; [informe 2007, Federal Income Taxes](https://www.sec.gov/Archives/edgar/data/1100663/000119312507104035/dncsr.htm); fechas [catálogo del emisor](https://www.blackrock.com/americas-offshore/en/products/product-list) | Sí; 3020 filas |
| HYG | Crédito corporativo HY +1x | 2007-04-04 / 2007-04-11 | RIC; [informe febrero 2008, Federal Income Taxes, p. 91](https://www.sec.gov/Archives/edgar/data/1100663/000119312508105554/dncsr.htm); fechas [emisor](https://www.ishares.com/us/literature/product-brief/ishares-high-yield-corporate-bond-etfs-product-brief-en-us.pdf) | Sí; 2953 filas |
| EMB | Soberano emergente USD +1x | 2007-12-17 / 2007-12-19 | RIC; [informe febrero 2008, Federal Income Taxes, p. 91](https://www.sec.gov/Archives/edgar/data/1100663/000119312508105554/dncsr.htm); fechas [emisor](https://www.blackrock.com/americas-offshore/en/products/product-list) | Sí; 2777 filas |
| BWX | Soberano internacional +1x | 2007-10-02 / 2007-10-11 | RIC; [prospecto del emisor 2007, Investment Objective y Non-Diversified Status](https://br.advfn.com/noticias/EDGAR/2007/artigo/22581139); fecha [emisor](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-bloomberg-international-treasury-bond-etf-bwx) | Sí; 2825 filas |
| EWA | Acciones Australia +1x | 1996-03-12 / 2007-01-03 | RIC; [informe 2007, Federal Income Taxes](https://www.sec.gov/Archives/edgar/data/930667/000119312507238657/dncsr.htm); fecha [emisor](https://www.ishares.com/us/products/etf-investments) | Sí; 3020 filas |
| SH | S&P500 diario -1x | 2006-06-19 / 2007-01-03 | RIC; [prospecto 2006](https://www.sec.gov/Archives/edgar/data/1174610/000119312506133844/d497.htm); fecha [emisor](https://prod.proshares.com/our-etfs/leveraged-and-inverse/sh) | Sí; 3020 filas |
| PSQ | Nasdaq100 diario -1x | 2006-06-19 / 2007-01-03 | RIC; mismo prospecto 2006; fecha y mandato [emisor](https://www.proshares.com/our-etfs/leveraged-and-inverse/psq) | Sí; 3020 filas |
| FXS | SEK/USD +1x | 2006-06-26 / sin precios Yahoo | GT; [10-Q 2006, Trust Overview](https://www.sec.gov/Archives/edgar/data/1353616/000089706906002042/cmw2329.htm) | Sí, dato pendiente; última negociación 2020-02-14 según [OCC 47036, 2020-05-22](https://infomemo.theocc.com/infomemos?number=47036) |
| UDN | Cesta monedas frente al USD +1x | 2007-02-20 / no descargado | Partnership; [10-Q junio 2018, Income Taxes](https://www.sec.gov/Archives/edgar/data/1371571/000095012318007518/udn-10Q_20180630.htm) | No: C4, PTP; no hace falta resolver las otras condiciones |
| SSO | S&P500 diario +2x | 2006-06-19 / no descargado | [Mandato y fecha del emisor](https://www.proshares.com/our-etfs/leveraged-and-inverse/sso) | No: C3, apalancado; fiscalidad no evaluada |
| CEW | Monedas emergentes +1x | 2009-05-06 / no descargado | [Anuncio del emisor 2009-05-06](https://www.wisdomtree.com/us/press-room/wisdomtree-and-dreyfus-launch-emerging-currency-fund-cew) | No: C1, posterior al corte; fiscalidad no evaluada |

Los admitidos descargados siguen activos en las fuentes consultadas; no se detectó
baja. FXS permanece en el universo elegible: `download_daily("FXS", "2007-01-03")`
devolvió 0 filas, HTTP 404, el 2026-10-22. El snapshot disponible tiene sesgo de
supervivencia y no reemplaza ese faltante por otro fondo. No habilita afirmar
representatividad de todos los ETFs existentes al corte.

Receta ejecutada desde la raíz, usando el destino persistente de entorno:

```bash
python -m quantagent.cli data snapshot create --name etf-ampliados-1d-2018-20261022 --symbols FXA,FXB,FXC,FXY,LQD,HYG,EMB,BWX,EWA,SH,PSQ --start 2007-01-03 --end 2018-12-31
python -m quantagent.cli data snapshot verify --name etf-ampliados-1d-2018-20261022
```

Creación: `Snapshot etf-ampliados-1d-2018-20261022 creado: 11 símbolos, 32687 filas`.
Fuente Yahoo Finance vía yfinance **1.7.0**, descarga **2026-10-22T21:12:02+00:00**.
El CLI recibe historia hasta hoy, recorta antes de escribir y congela un nombre nuevo;
no conserva velas de 2019 en adelante. OHLCV ajustados por splits, `adj_close`
separado para dividendos; sesiones de bolsa, sin relleno ni interpolación.
Los Parquet y `manifest.json` viven únicamente en
`$QUANTAGENT_SNAPSHOT_DIR/etf-ampliados-1d-2018-20261022/`.
El [manifiesto versionado](../tests/fixtures/snapshots/etf-ampliados-1d-2018-20261022.json)
registra SHA256 por archivo, filas y extremos reales; `verify` contrasta ambos.
Verificación real: `OK 11 símbolos, 32687 filas` (exit 0). `snapshot check` salió 0:
sin sesiones ausentes ni OHLC incoherentes; EMB tiene una fila con volumen cero
y EWA un salto ajustado >20% el 2008-10-13. Se conservaron sin corrección.
Los 16 archivos preexistentes conservaron el SHA256 conjunto antes/después
`741a4c1fc144864b45544287c65614778f74f9175e913cc0e5c6f5bd58f5dcd7`
(concatenación ordenada de ruta relativa y bytes, excluyendo el directorio nuevo).

Todo precio almacenado es historia observada del ETF; ningún proxy se implementó.
Aplican las convenciones del [empalme D-2b](#empalme-de-proxies-de-investigación-quantagent-e4jc82).
FXY comienza en febrero; HYG, BWX y EMB también dejan faltantes iniciales reales.
No se extendieron hacia atrás ni se calcularon métricas, señales o resultados de
estrategias, tampoco en 2019–2022 ni en la reserva 2023+.

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
