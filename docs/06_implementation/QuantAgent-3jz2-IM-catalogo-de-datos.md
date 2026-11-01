# QuantAgent-3jz2 · Catálogo de datos

- **Ticket:** QuantAgent-3jz2, B2 del lote M-cuenta, epic xqd7.
- **Rama:** `m-cuenta-b2`; entrega sin commit, pendiente de revisión e integración humana.

## Implementación

Ocho YAML en `catalogo/datos/` agrupan los once manifiestos versionados por familia:
precios ETF, precios de acciones, composición, fundamentales, macro, consenso EPS,
factores y futuros de pysystemtrade. Registran campos conocidos, licencia, disponibilidad
temporal, patrones de carpeta y lectores comprobados en código. `_no-son-datos.yaml`
identifica resultados, registros, temporales y directorios técnicos observados.

`quantagent/data/catalogo.py` carga los YAML y verifica cobertura sin abrir datos.
El test valida esquema, identidad archivo/id y correspondencia exacta de manifiestos.
Con datos locales exige cobertura en ambos sentidos; sin carpetas hace skip explícito.
Los casos con `tmp_path` prueban carpeta huérfana, patrón ausente y aislamiento de
fuentes con regla de borrado, sin declarar un dataset Sharadar inexistente.

## Límites y relación con el ADR

Se sigue el [ADR de almacenamiento](../04_decisions/QuantAgent-aj1s-DC-almacenamiento-de-datos.md).
El catálogo describe la realidad existente: fundamentales también tiene un congelado
diario de paper (`2026-11-01`, manifiesto sólo local), aunque el ADR fija congelado
mensual. No se modifica ese comportamiento. Consenso conserva fotos diarias sin
manifiesto. Factores y pysystemtrade están documentados pero ausentes del checkout
principal al 2026-11-01: `datos_esperados: false` evita exigir su descarga.
No se copiaron datos ni se agregaron dependencias; PyYAML ya está declarado y disponible.

## Validación y handoff

Desde la raíz, con el entorno existente en `PATH` y `PYTHONPATH=.`:

```bash
python -m pytest tests/test_catalogo_datos.py -q
QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data python -m pytest tests/test_catalogo_datos.py -q
python -m pytest tests/docs -q
git diff --check
```

El test local usa la marca `snapshot` para conservar la variable de entorno ante la
fixture global de aislamiento. Sin datos: **4 passed, 1 skipped**. Con el checkout
principal: **5 passed**. Documentación: **3 passed, 1 failed**; el fallo preexistente
es el enlace faltante al ADR `QuantAgent-aj1s` en `docs/manual/decisions.md`, también
ausente en HEAD. No se modifica el manual en este ticket. No se encontraron tests
que enumeren los directorios de la raíz; se corrieron los cuatro de `tests/docs/`.

Beads no pudo consultar ni actualizar el ticket: `bd show QuantAgent-3jz2` devolvió
`sqlite3: unable to open database file`. Pendiente actualizar su estado al integrar.
Cambios sin commit por instrucción del usuario; mensaje sugerido:
`feat(QuantAgent-3jz2): catálogo de datos y cobertura de snapshots`.
