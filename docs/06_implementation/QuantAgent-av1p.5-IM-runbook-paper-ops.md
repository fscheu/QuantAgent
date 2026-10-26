# QuantAgent-av1p.5 — Runbook de operación de paper trading en la VM

**Issue:** QuantAgent-av1p.5
**Fecha:** 2026-10-26
**Estado:** propuesta. Nada de lo que está en §3 y §4 (credenciales, unidades systemd) está instalado ni ejecutado; cada paso marcado **[OK Fede]** espera su aprobación explícita.
**Base:** ítems 15 y 16 de [QuantAgent-L21-PL-paper-alpaca-m4.md](../02_planning/QuantAgent-L21-PL-paper-alpaca-m4.md); QuantAgent-av1p.4 y QuantAgent-e4jc.13 (cerrados, en main).
**Ventana:** primer ciclo lunes 2026-11-02, unos dos meses (hasta ~2027-01-04).

## 1. Qué es y qué no es

Paper es una prueba del sistema, no una evaluación realista (DC R01). Se mira si las piezas funcionan sobre datos reales sin intervención: pesos iguales a la sombra salvo redondeo, ninguna orden rechazada sin explicar, dividendos bien tratados. Si algo no coincide, se frena y se investiga. Ningún número de estos dos meses promueve una estrategia a dinero real.

Carteras: `hipotesis/cartera-transversal.yaml` y `hipotesis/cartera-nucleo-curva.yaml`. Solo la segunda necesita `FRED_API_KEY` (pierna curva-pendiente). La pierna curva depende de macro vigente (QuantAgent-2j54 cerrado); si el dato venció, esa pierna no opera y lo reporta.

## 2. Decisión previa: cuentas Alpaca (bloqueante) **[OK Fede]**

El pedido original dice "una cuenta, con DB y proceso independientes por cartera". Con una sola cuenta Alpaca real, **no funciona con el código actual**: `TradingScheduler._process_paper_portfolio` pisa `portfolio.cash` y `portfolio.positions` con `broker.get_account()` y `broker.get_positions()` de la cuenta entera. Dos procesos contra la misma cuenta verían las posiciones y la caja del otro como propias, y las dos YAML comparten activos (SPY, QQQ, etc.), así que se pelearían por los mismos símbolos. Además la reconciliación de posiciones (y62) compararía la DB de una cartera contra las posiciones de ambas.

Lectura que propongo: **un login de Alpaca con dos cuentas paper separadas**, una por cartera, cada una con su par de claves. Es lo que Fede decidió el 2026-10-24 (ver `docs/loop/particiones/QuantAgent-av1p.4.md`; Alpaca permite hasta tres cuentas paper por usuario). Si en cambio se quiere una sola cuenta, hace falta código nuevo (contabilidad por cartera dentro de la cuenta) y no llega para el 11-02.

Hace falta que Fede cree la segunda cuenta paper en el dashboard de Alpaca y cargue ambas con el mismo capital inicial (por ejemplo 100 000 USD, igual que `TRADING_INITIAL_CASH`).

## 3. Credenciales **[OK Fede]**

Hoy `~/secrets/alpaca.env` tiene `APCA_API_KEY_ID` y `APCA_API_SECRET_KEY` (los que lee la fuente de datos `quantagent/data/sources/alpaca.py`). El broker (`strategy/assembler.py`) lee otros nombres: `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`. systemd no renombra variables, así que cada archivo por cartera debe traer los dos juegos con los mismos valores.

Propuesta, un archivo por cartera, modo 600, fuera del repo:

```
~/secrets/alpaca-cartera-transversal.env
~/secrets/alpaca-cartera-nucleo-curva.env
```

Contenido de cada uno (valores de la cuenta paper correspondiente):

```
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
ALPACA_API_KEY=...        # mismo valor que APCA_API_KEY_ID
ALPACA_SECRET_KEY=...     # mismo valor que APCA_API_SECRET_KEY
```

`~/secrets/fred.env` (`FRED_API_KEY`) se agrega como segundo `EnvironmentFile` solo en la unidad de `cartera-nucleo-curva`.

El código rechaza `alpaca_paper` si la URL no es `https://paper-api.alpaca.markets`, así que una clave live no opera por esta ruta. Igual: verificar en el dashboard que ambos pares son de cuentas **paper** antes de cargarlos.

Claude no lee ni copia el contenido de `~/secrets/`; el archivo lo arma Fede (o autoriza que lo arme Claude sin imprimir valores).

## 4. Despliegue **[OK Fede]**

### 4.1 Código congelado

El checkout `~/repos/projects/QuantAgent` lo usa el loop (cambia de rama y de commit). Un servicio de dos meses no puede correr sobre eso. Propuesta: un clon aparte fijado a un tag.

```bash
git clone ~/repos/projects/QuantAgent ~/paper-ops/QuantAgent
cd ~/paper-ops/QuantAgent && git checkout -b paper-ops <commit-de-main-con-docs-mergeados>
git tag paper-2026-11-02
```

El venv es el del checkout principal (`/home/azureuser/repos/projects/QuantAgent/.venv`), instalado en modo editable apuntando a ese checkout. Para que importe el clon, las unidades fijan `WorkingDirectory` y `PYTHONPATH` al clon. Verificación antes de arrancar:

```bash
cd ~/paper-ops/QuantAgent && PYTHONPATH=$PWD /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -c "import quantagent; print(quantagent.__file__)"
```

Debe imprimir una ruta bajo `~/paper-ops/`. Si imprime la del checkout principal, no arrancar.

Los datos viven en `~/paper-ops/QuantAgent/local-data/`: `<cartera-id>.db` (SQLite) y `<cartera-id>.sombra.json`, un par por cartera (`apps/paper_trading.py:_build_portfolio_scheduler`). Aplica `alembic upgrade head` a cada DB si el esquema no se crea solo con `create_all` (la columna `alpaca_order_id` es de av1p.4.2): verificar en el ensayo (§6).

### 4.2 Un ciclo diario, no un daemon

El scheduler del código usa `IntervalTrigger` por horas y corre inmediato al arrancar. Para una cartera diaria sobre barras cerradas es más simple y robusto un **timer de systemd que lanza un proceso `--run-once`** de lunes a viernes, después del cierre de NYSE. Cada ciclo: refresca órdenes pendientes (`refresh_rebalance_orders`), sincroniza caja y posiciones con Alpaca, calcula pesos con la última sesión cerrada, rebalancea y actualiza la sombra. Si el proceso muere, el próximo día arranca limpio y las órdenes pendientes quedan en la DB por su ID de Alpaca.

Horario: **23:30 UTC (20:30 ART), lunes a viernes.** NYSE cierra 21:00 UTC desde el 2026-11-01 (fin del horario de verano de EE.UU.), y 18:00 UTC en los cierres tempranos (2026-11-27 y 2026-12-24); 23:30 cubre ambos y da margen para que la barra diaria esté publicada. Las órdenes que salen a esa hora son de mercado y llenan en la apertura del día siguiente; el ciclo de ese día las refresca primero.

Unidad (plantilla; `%i` = id de la cartera):

```ini
# ~/.config/systemd/user/paper-cartera@.service
[Unit]
Description=QuantAgent paper: ciclo diario de %i

[Service]
Type=oneshot
WorkingDirectory=/home/azureuser/paper-ops/QuantAgent
Environment=PYTHONPATH=/home/azureuser/paper-ops/QuantAgent
Environment=NUMBA_CACHE_DIR=/tmp/numba-cache
Environment=TRADING_SCHEDULER_ENVIRONMENT=paper
EnvironmentFile=/home/azureuser/secrets/alpaca-%i.env
ExecStart=/home/azureuser/repos/projects/QuantAgent/.venv/bin/python apps/paper_trading.py --portfolio hipotesis/%i.yaml --run-once --enable
TimeoutStartSec=30min
```

Para `cartera-nucleo-curva` se agrega un drop-in con `EnvironmentFile=/home/azureuser/secrets/fred.env`.

```ini
# ~/.config/systemd/user/paper-cartera@.timer
[Unit]
Description=Ciclo paper diario de %i (lun-vie tras el cierre NYSE)

[Timer]
OnCalendar=Mon..Fri 23:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

Activación (solo con OK de Fede):

```bash
systemctl --user daemon-reload
systemctl --user enable --now paper-cartera@cartera-transversal.timer paper-cartera@cartera-nucleo-curva.timer
```

Alternativa en cron si se prefiere: una línea por cartera con el mismo comando y `set -a; . ~/secrets/alpaca-<id>.env; set +a`. Se pierde `Persistent=true` (recuperar un ciclo perdido por reinicio de la VM) y el log en el journal; no la recomiendo.

El linger del usuario ya está habilitado (sobrevive logout), como en `hermes-gateway.service`.

## 5. Digest diario por Telegram

El digest sale del CLI existente (`quantagent trading digest`, QuantAgent-b41.2) y se entrega con `hermes send`, que reutiliza las credenciales de Telegram de Hermes **sin LLM y sin tocar su configuración**. Se envía a las 23:50 UTC, veinte minutos después del ciclo.

Contenido útil por cartera, en este orden:

1. `quantagent trading digest --date <hoy> --db local-data/<id>.db --portfolio <id>`: heartbeat del scheduler (alerta si no hay ciclo completado hace más de un día hábil), operaciones y P&L, posiciones, avisos.
2. Chequeos e4jc.13 (§7).
3. Reconciliación con Alpaca (§7, chequeo 4).

Unidad:

```ini
# ~/.config/systemd/user/paper-digest@.service
[Unit]
Description=QuantAgent paper: digest diario de %i

[Service]
Type=oneshot
WorkingDirectory=/home/azureuser/paper-ops/QuantAgent
Environment=PYTHONPATH=/home/azureuser/paper-ops/QuantAgent
Environment=NUMBA_CACHE_DIR=/tmp/numba-cache
ExecStart=/bin/bash -c '/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m quantagent.cli trading digest --date "$(date -u +%%F)" --db local-data/%i.db --portfolio %i | /home/azureuser/.local/bin/hermes send -t telegram -s "Paper %i"'
```

con un `paper-digest@.timer` análogo: `OnCalendar=Mon..Fri 23:50:00 UTC`. Si el CLI falla, el pipe no manda nada y el silencio es una señal; por eso en §7 se pide un chequeo de "ausencia de digest".

## 6. Ensayo previo (antes del 11-02) **[OK Fede]**

No hay modo "sin órdenes": cualquier `--run-once` manda órdenes reales a la cuenta paper. El ensayo es el primer ciclo real, hecho a mano y mirando el resultado, antes de armar el timer:

1. Clon y tag (§4.1), claves cargadas (§3), verificación de importación.
2. Una cartera por vez, un día de mercado cualquiera (por ejemplo viernes 2026-10-30 tras el cierre): correr el comando de `ExecStart` a mano.
3. Verificar: sale con código 0; hay una fila en `scheduler_heartbeats` con `completed`; el archivo `local-data/<id>.sombra.json` tiene la entrada de la fecha; las órdenes aparecen en Alpaca con el mismo ID que en la DB; la caja no baja del 5 % del equity (§8).
4. Al día siguiente, correr de nuevo: las pendientes se refrescan, no se duplican, y el digest manual (`trading digest ...`) muestra los fills.
5. Si el esquema de la DB quedó atrás (`no such column alpaca_order_id`), correr alembic sobre esa DB antes de seguir y anotarlo acá.

Cualquier cosa rara en el ensayo se reporta a Fede antes de activar los timers. Las posiciones del ensayo se dejan: son la línea de base real del 11-02.

## 7. Chequeos diarios (e4jc.13) y alertas

Los tres chequeos existen como funciones en `quantagent/fabrica/chequeos_paper.py` (`comparar_pesos`, `reportar_rechazos`, `reportar_dividendos`) y como comando solo con archivo de entrada: `quantagent trading digest <archivo.json>` con las claves `paper_weights`, `shadow_weights`, `orders`, `dividends`; sale con código 1 si hay diferencias, rechazos sin explicación o dividendos sin tratamiento. **El modo `--date --db` del digest no los ejecuta**, y nada arma ese JSON desde la DB y la sombra. Es el hueco principal de esta operación (§10, ticket de seguimiento).

Hasta que exista, el chequeo diario es manual-asistido (Claude lo corre con Fede, o Fede mira el digest) sobre estos cuatro puntos:

| # | Chequeo | Cómo | Condición de frenar |
|---|---|---|---|
| 1 | Pesos paper contra sombra | `leer_sombra("local-data/<id>.sombra.json", "<id>", fecha)` contra pesos reales (posición × precio / equity) con `comparar_pesos` (tolerancia 1e-4) | Diferencia mayor que el redondeo de acciones enteras |
| 2 | Rechazos | Órdenes con estado rechazado en la DB, `reportar_rechazos` | Cualquier rechazo sin motivo en `comment` |
| 3 | Dividendos | `reportar_dividendos` sobre los dividendos del período | Dividendo sin tratamiento explicado |
| 4 | Reconciliación | `DATABASE_URL=sqlite:///.../local-data/<id>.db` + claves de esa cuenta, `quantagent trading reconciliar` | Cualquier divergencia entre DB local y Alpaca |

Además, tres alertas que el digest no cubre:

- **Ciclo "sano" que no hizo nada.** `entradas_cartera` devuelve `None` (sin barra diaria completa, historia corta, dato faltante) y el ciclo termina `completed` sin operar. Verificar a diario que la sombra tiene entrada de la fecha de la última sesión cerrada; si no, investigar el proveedor de datos aunque el heartbeat esté verde.
- **Ausencia del digest.** Si a las 00:10 UTC no llegó el mensaje de una cartera, el ciclo o el envío fallaron: `journalctl --user -u paper-cartera@<id> -u paper-digest@<id> --since today`.
- **Macro vencida** en `cartera-nucleo-curva`: debe figurar como bloqueo de la pierna, no como cartera en efectivo silenciosa.

### Qué hacer al frenar

```bash
systemctl --user stop paper-cartera@<id>.timer paper-digest@<id>.timer   # solo esa cartera, o ambas
```

No se cierran posiciones ni se editan la DB o la sombra a mano. Se anota el hallazgo en el ticket, se investiga con las DB y los logs intactos, y se reanuda recién con la causa entendida (decisión de Fede).

## 8. Reserva de 5 % de efectivo

Está fija en el código: `StrategyAssembler.build_components` pasa `cash_reserve_pct=0.05` al `OrderManager` en todo entorno que no sea backtest (`quantagent/strategy/assembler.py:245`). No hay variable de entorno para cambiarla. Consecuencias a vigilar:

- La caja de cada cuenta paper no debe bajar del 5 % del equity. Chequeo diario: `cash / equity` en `broker.get_account()`; por debajo de 5 % es hallazgo (el ensayo del §6 lo verifica por primera vez).
- Los pesos efectivos suman como máximo 95 %. Si `comparar_pesos` muestra una diferencia sistemática del orden de la reserva entre sombra y paper, **no se interpreta como error ni se corrige a mano**: se reporta, porque es una decisión sobre cómo se define la sombra (con o sin reserva), no un bug obvio. No lo pude confirmar leyendo el código; lo muestra el ensayo.

## 9. Fin de la ventana

Aproximadamente el 2027-01-04 (o antes, si se frena): desactivar timers, copiar `local-data/` a `~/backups/paper-<fecha>/`, dejar las posiciones sin liquidar hasta que Fede decida, y un informe corto en el ticket: ciclos completados, cantidad de frenadas y sus causas, y qué piezas del sistema fallaron. Sin conclusiones sobre rendimiento de estrategias.

## 10. Seguimientos propuestos (no incluidos en este PR)

1. **Chequeo diario automático** (hueco de §7): comando o paso del digest que lea DB y sombra, ejecute los tres chequeos de e4jc.13 y falle con código ≠ 0. Ticket de código pequeño, hijo de QuantAgent-av1p.5. Sin esto la operación depende de mirar a mano.
2. Entrada del digest para "sombra sin entrada de la fecha" (alerta de ciclo vacío).
3. Si se quiere una sola cuenta de verdad, ticket de diseño aparte (contabilidad por cartera dentro de la cuenta).

## 11. Aprobaciones pendientes de Fede

- [ ] Un login de Alpaca con dos cuentas paper, una por cartera (§2), o la alternativa.
- [ ] Crear/cargar `~/secrets/alpaca-<id>.env` (§3).
- [ ] Clon `~/paper-ops/QuantAgent` y tag (§4.1).
- [ ] Instalar las unidades y timers de §4.2 y §5.
- [ ] Ensayo manual de §6 (manda órdenes a la cuenta paper).
- [ ] Activar los timers para el 2026-11-02.
