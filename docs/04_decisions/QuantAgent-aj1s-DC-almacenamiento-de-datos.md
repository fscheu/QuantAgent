# QuantAgent-aj1s · ADR: almacenamiento de datos

- **Ticket:** QuantAgent-aj1s (lote `lote/m-cuenta`, frente B1).
- **Fecha de decisión:** 2026-10-28.
- **Estado:** Aceptada por Fede (D1, D2 y D3 de M-datos).

## Contexto

Los datos históricos de mercado viven en snapshots Parquet con manifiesto y hashes; los registros del embudo, en CSV; y el estado operativo, en Postgres. Los snapshots inmutables permiten repetir un backtest aunque el proveedor recalcule su historia. El universo de acciones y fundamentales de M5 traerá más volumen y consultas sobre lo que se sabía en cada fecha.

La revisión crítica aceptada por Fede simplificó la propuesta inicial: para 12–50 ETFs alcanza con pandas y una descarga completa mensual. No hace falta construir ahora una capa viva con eventos corporativos para recalcular ajustes.

## Opciones

- **Todo en carpetas:** simple y reproducible, pero las consultas sobre lo que se sabía en una fecha requieren lógica propia y escalan peor con el universo de acciones.
- **Todo en Postgres:** unifica las consultas, pero no favorece las lecturas masivas de historia ni conserva por sí solo la inmutabilidad del snapshot.
- **Esquema mixto:** Parquet para históricos, DuckDB para consultas cuando llegue el universo de acciones y Postgres para operación y registros del embudo cuando la web los necesite. Es la opción elegida.
- **SQLite para los registros:** descartada frente a Postgres por la escritura desde varios procesos, las migraciones con tipos propios de Postgres y la base de paper ya operativa allí.
- **TimescaleDB o ClickHouse:** descartadas por ahora; agregan un servicio para una escala que todavía no tenemos.

## Decisión

1. **D1: almacenamiento mixto.** Los históricos de mercado y fundamentales quedan en **Parquet inmutable con manifiesto, en carpetas por fuente y fecha**. DuckDB se incorpora recién cuando llegue el universo de acciones; para ETFs se sigue con pandas. Los registros del embudo pasan a Postgres cuando la web lo necesite.
2. **D2: congelado mensual.** Una vez por mes se vuelve a bajar la historia completa de cada fuente y se congela un snapshot nuevo, sin modificar los anteriores. Un activo nuevo entra al próximo congelado. Paper sigue leyendo los precios del día desde Alpaca y guarda sus decisiones en la base. La foto del consenso de EPS se guarda a diario porque el proveedor no conserva esa historia.
3. **D3: Postgres, no SQLite.** El estado operativo permanece en Postgres; los registros del embudo se incorporarán allí cuando corresponda.

La organización por fuente debe permitir cumplir la licencia de Sharadar: **borrar una fuente equivale a borrar su carpeta**. Si se deja de pagar Sharadar, sus datos deben borrarse en 30 días, según la condición registrada en el roadmap.

## Consecuencias

Se conserva la reproducibilidad de los backtests y se evita mantener por ahora una capa diaria de precios con ajustes propios. Los CSV del embudo siguen hasta que la web justifique su migración. La baja de una fuente debe respetar su licencia, aunque eso limite la conservación de sus snapshots.

**Fuera del alcance de este ticket:** implementar DuckDB, migrar el embudo a Postgres, construir el esquema de dos capas (histórica y viva) y decidir la contratación o continuidad de Sharadar. Este ADR registra las decisiones; no las implementa.

## Fuentes

- [Roadmap posterior a M5, §4: M-datos y decisiones D1/D2/D3](../02_planning/2026-10-28_roadmap_post_m5.md#4-m-datos-dónde-guardar-los-datos-de-mercado-y-fundamentales).
- [Revisión crítica del roadmap: M-datos (§2) y cambio 5 aceptado (§4)](../02_planning/2026-10-28_revision_critica_roadmap_post_m5.md).
