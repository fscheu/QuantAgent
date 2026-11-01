# Runtime único de paper-ops (plan de migración para el lunes 2026-11-02)

Estado: **propuesto, sin aplicar**. Pedido de Fede (2026-11-01): una sola versión productiva de QuantAgent, asociada a un tag, que corra las dos carteras M4 (nucleo-curva, transversal) y M5 contra Alpaca paper, con los jobs diarios y semanales, y con un directorio de datos único y separado del de desarrollo. Cada mejora de la semana se despliega con un tag nuevo. Cambia el ADR `ADR-runtime-paper-deploy-por-tags` solo en el punto de los datos (antes cada clon traía su `local-data/`).

## Estado de hoy

| Pieza | Dónde está hoy |
|---|---|
| Clon M4 + web | `~/paper-ops/QuantAgent` (tag `paper-2026-10-30-web2`), datos en `QuantAgent/local-data` |
| Clon M5 | `~/paper-ops/QuantAgent-m5` (tag `paper-m5-2026-11-02`), datos en `QuantAgent-m5/local-data` |
| Timers | M5 activos (`m5-semanal`, `m5-diario`, `m5-pre-apertura`, `m5-digest`); M4 (`paper-cartera@`, `paper-digest@`) apagados desde el 10-30 |
| Web | `quantagent-api.service` corre desde el clon M4; `~/web-ops/maestro.db` guarda la ruta absoluta de las bases de las dos carteras M4 |
| Snapshots | solo en el directorio de desarrollo (`~/repos/projects/QuantAgent/local-data`) |

## Destino

```
~/paper-ops/
  QuantAgent/      clon único; solo `git checkout <tag>` (nadie edita)
  venv/            entorno único
  local-data/      DATO ÚNICO: bases SQLite de las 3 carteras, CSV de M5, sombra.json, snapshots
  backups/
```

Todas las unidades usan `WorkingDirectory=%h/paper-ops` (así las rutas relativas `local-data/...` del código caen en el dato único), `PYTHONPATH=%h/paper-ops/QuantAgent`, `QUANTAGENT_SNAPSHOT_DIR=%h/paper-ops/local-data` y rutas absolutas al clon para `apps/paper_trading.py` e `hipotesis/*.yaml`. El clon `QuantAgent-m5` se retira (se conserva renombrado como respaldo una semana). El `local-data/` que queda dentro del clon no se usa.

El directorio de desarrollo (`~/repos/projects/QuantAgent/local-data`) no se toca: de ahí solo se **copian** los snapshots que paper necesita.

## Qué lotes entran el lunes

- **Lunes: `lote/m5-paper` (#719).** Es lo que necesita el paper de M5. CI verde.
- **Después, con su propio tag:** `lote/m-web-2` (#743) y `lote/cuenta-chica` (#750), que siguen en curso. Cuando cada uno cierre y pase CI se mergea y se despliega con tag nuevo; el deploy es `checkout` del tag y reinicio de lo que cambie (web y/o jobs), fuera de horario de mercado. No bloquean el lunes.
- Mientras tanto la web corre desde el mismo clon: pasa al tag nuevo y por eso no gana ni pierde nada de M-web hasta que su lote entre.

## Pasos (domingo y lunes antes de las 13:30 UTC, pre-apertura de M5)

Cada paso con respaldo previo; ninguno envía órdenes.

1. **Hoy (sin impacto):** mergear #719 a `main` con CI verde; crear tag `paper-2026-11-02` sobre `main`. Armar `~/paper-ops/local-data-nuevo/` copiando los snapshots (`fundamentales-sp500-2026-10`, `fundamentales-sp500-2026-11-01`, `acciones-sp500-2026-10b`, `sp500-2026-10`, `etf-m5-1d-2026-10`, `macro-m5-2026-10`, más `etf-1d-2026-10-amplio` y `macro-2026-10` por si las carteras M4 los leen). Verificar hashes.
2. **Ventana de cambio (domingo, sin órdenes pendientes):**
   a. Parar timers M5 y `quantagent-api`. Backup en `~/paper-ops/backups/20261101-runtime-unico/` de ambos `local-data/`, `maestro.db` y las unidades.
   b. `~/paper-ops/QuantAgent`: `git fetch --tags && git checkout paper-2026-11-02`. Verificar `pip check` del venv contra `pyproject.toml` (reinstalar con `pip install -e .` solo si falta algo).
   c. Mover al dato único: `cartera-nucleo-curva.db`, `cartera-transversal.db`, sus `*.sombra.json` (del clon M4) y `m5.db`, `m5-regimen-semanal.csv`, `m5-narrativa.csv` (del clon M5); copiar ahí los snapshots. Renombrar `QuantAgent-m5` a `QuantAgent-m5.retirado`.
   d. Actualizar en `maestro.db` las dos filas de `accounts`: `database_url` pasa de `.../QuantAgent/local-data/<id>.db` a `.../paper-ops/local-data/<id>.db`. Registrar la cuenta M5 (hoy no figura en la web) con su base `m5.db`.
   e. Actualizar las unidades (se muestra el diff exacto antes de aplicar): `m5-*`, `paper-cartera@`, `paper-digest@` y `quantagent-api` con el `WorkingDirectory` y las variables de arriba; unidad nueva `m5-fundamentales` (viernes 22:30 UTC, `data fundamentales refrescar`, timeout 60 min) antes de `m5-semanal`. `daemon-reload`.
3. **Verificación sin órdenes:** `quantagent trading reconciliar` en las tres carteras; `m5-correr --modo semanal` manual (solo escribe CSV) y comparar contra el ensayo en seco del 11-01; `curl` a la web y ver las 2 + 1 carteras.
4. **Encender:** `systemctl --user enable --now` de los timers M4 (`paper-cartera@cartera-nucleo-curva`, `...-transversal`, `paper-digest@...`) y reiniciar `quantagent-api`. Los de M5 ya están activos.
5. **Lunes 13:30 UTC:** corre `m5-pre-apertura`. Revisión del digest y de la reconciliación (checklist de M5-55f).

## Deploys de la semana

`git tag paper-AAAA-MM-DD` sobre `main` (PM o Fede, tras revisar) → `git -C ~/paper-ops/QuantAgent checkout <tag>` → reiniciar solo lo que cambió, fuera de horario de mercado y sin órdenes pendientes. El dato no se toca. Si algo falla: volver al tag anterior; el dato único no depende del tag, pero cualquier migración de base se hace con respaldo previo (ADR, punto 4).

## Riesgos

- Mover las bases mientras un timer está corriendo: por eso se paran antes. Una base movida con el `maestro.db` sin actualizar deja a la web sin cartera: se verifica en el paso 3.
- Las carteras M4 llevan apagadas desde el 10-30: al encenderlas con `Persistent=true` pueden disparar un ciclo atrasado. Se revisa con `list-timers` antes de habilitar.
- Dependencias del venv respecto del tag nuevo (ya se agregó `six`).
- Alpaca: M4 y M5 usan cuentas distintas (`alpaca-cartera-nucleo-curva.env`, `...transversal.env`, `...m5.env`); no se mezclan.
