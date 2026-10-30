# QuantAgent-e452.1 · Arquitectura de la web

- **Ticket:** WEB-01, `QuantAgent-e452.1`.
- **Estado:** propuesta; las decisiones del ADR asociado quedan a confirmar por Fede, salvo cortos, ya decidido por Fede el 2026-10-30.
- **Alcance:** arquitectura de consulta para Inicio y Cuenta. Embudo queda diferido hasta que exista su registro de datos; no se habilitan operaciones desde la web.
- **Referencias:** [ADR de la web](../04_decisions/QuantAgent-e452.1-DC-web.md), [design system](./web-design-system/README.md), [roadmap M-web](../02_planning/2026-10-28_roadmap_post_m5.md#3-m-web-aplicación-web-que-reemplaza-streamlit).

## Piezas

- **API:** FastAPI en Python, con modelos de respuesta Pydantic. Es la única pieza que accede a las bases y transforma el modelo interno en el contrato HTTP.
- **Interfaz:** React, TypeScript y Vite. Consume endpoints `GET` y presenta los tipos del design system. El bundle compilado se sirve en la misma VM.
- **Persistencia:** Postgres es la fuente para órdenes, trades, posiciones, señales y estado operativo disponible en los modelos SQLAlchemy. DuckDB se consulta para series y snapshots analíticos que M-datos publique allí. La API consulta cada fuente según el dato; no replica ni escribe datos de trading. Las tablas de snapshots/eventos que todavía no existen requieren sus tickets propios.

## Identidad de cuentas

Propuesta: la cuenta visible representa una estrategia/bucket operativo dentro de un entorno `paper` o `live`. El `id` estable combina ambos, por ejemplo `paper-m4`; el entorno no reemplaza al bucket. La configuración declara las cuentas esperadas aunque todavía no tengan datos. Para una cuenta aún no abierta o financiada, `AccountSummary` devuelve `status: 'no_data'` y nunca valores cero ficticios.

El modelo actual no permite reconstruir toda esa identidad de forma uniforme: `Order`, `Trade` y `Position` tienen `environment`; `trade_intents` tiene `bucket_id`, pero la orden no. WEB-02/03 deben aportar el mapeo explícito entre bucket, entorno y credencial/cuenta Alpaca antes de publicar varias cuentas. Las cuentas Alpaca paper/live son la conexión externa de ese bucket, no un sustituto del bucket en el contrato. Se preservan las diferencias paper/live y M4/M5; no se agregan totales entre cuentas.

## Lectura y contrato

La API abre sesiones de solo lectura para consultas en Postgres y conexiones de lectura para DuckDB. SQLAlchemy usa `DATABASE_URL` para el motor relacional. Las consultas filtran explícitamente por entorno y, cuando esté disponible, por bucket; la falta de ese vínculo se resuelve en el backend mediante el mapeo acordado, no mezclando filas por símbolo. DuckDB sirve datos históricos/analíticos, no órdenes ni posiciones operables.

Pydantic normaliza los nombres y valores del modelo SQLAlchemy al contrato documentado en `web-design-system/api-types.ts`. Fechas y horas son ISO 8601 UTC con `Z`; la consulta diaria de órdenes usa fecha calendario de Nueva York. Los endpoints previstos en `PANTALLAS.md` son de lectura. La app no invoca el SDK ni la API de Alpaca.

## Despliegue y acceso

La API y el scheduler corren como servicios separados en la misma VM y comparten configuración y acceso a las bases; el scheduler conserva la responsabilidad de actualizar datos. Propuesta de puertos: API en `127.0.0.1:8000`; bundle web servido por el proceso web en `127.0.0.1:8080`. Tailscale Serve publica la web a la red privada y enruta `/api` a la API; ninguno de los dos puertos escucha en interfaces públicas. No se requiere usuario/contraseña propia: Tailscale autentica quién puede llegar a la VM. La lista de cuentas queda limitada a las declaradas en configuración; la autenticación de red no concede una cuenta Alpaca ni amplía el conjunto de cuentas mostrado.

## Límite operativo

La web y su API son de solo lectura: nunca colocan, cancelan ni modifican órdenes, posiciones, stops o configuración del broker. Tampoco exponen comandos de trading ni credenciales al navegador. Cualquier operación futura requeriría otro diseño y una decisión explícita; no forma parte de WEB-01.
