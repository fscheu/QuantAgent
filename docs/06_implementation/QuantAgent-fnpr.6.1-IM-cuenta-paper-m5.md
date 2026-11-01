# QuantAgent-fnpr.6.1 — Cuenta paper M5

## Perfil y cableado

Las hipótesis `m5-*` usan capital inicial de USD 3.000, riesgo por operación de 0,25 %, máximo de 10 % por posición y heat máximo de 2 %. El sizing usa la regla `riesgo`; no se aplica filtro R/R. El heat está fijado en el perfil M5 y el default global de M4 sigue apagado.

Al sincronizar posiciones paper, el scheduler agrega `stop_price` desde una orden stop de venta abierta en Alpaca. Si no existe, usa el `stop_loss` del intent más reciente para ese símbolo. Si no hay ninguno, omite la clave. Un error al consultar órdenes se registra como warning y la sincronización continúa sin stops.

El proceso M5 carga `~/secrets/alpaca-cartera-m5.env` y completa en memoria los nombres `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` y `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. El archivo debe contener un juego de cada par; no se guardan claves en el repositorio ni en los logs. La base se crea en `local-data/m5.db` (el id de la hipótesis M5 será `m5`; el YAML lo define M5-51).

## Unidad propuesta, sin instalar

Plantilla basada en [el runbook av1p.5](QuantAgent-av1p.5-IM-runbook-paper-ops.md). Requiere aprobación de Fede antes de instalar o activar:

```ini
# ~/.config/systemd/user/paper-m5.service
[Unit]
Description=QuantAgent paper: ciclo diario M5

[Service]
Type=oneshot
WorkingDirectory=/home/azureuser/paper-ops/QuantAgent
Environment=PYTHONPATH=/home/azureuser/paper-ops/QuantAgent
Environment=NUMBA_CACHE_DIR=/tmp/numba-cache
Environment=TRADING_SCHEDULER_ENVIRONMENT=paper
EnvironmentFile=/home/azureuser/secrets/alpaca-cartera-m5.env
ExecStart=/home/azureuser/repos/projects/QuantAgent/.venv/bin/python apps/paper_trading.py --portfolio hipotesis/m5.yaml --run-once --enable
TimeoutStartSec=30min
```

```ini
# ~/.config/systemd/user/paper-m5.timer
[Unit]
Description=Ciclo paper M5 diario (mar-sab 08:00 UTC)

[Timer]
OnCalendar=Tue..Sat 08:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

No ejecutar `daemon-reload`, `enable` ni `start` como parte de este cambio.

## Verificación en la VM

Con el secreto creado por Fede, checkout congelado y timer aún inactivo:

1. Verificar que el import apunta al checkout de operación: `PYTHONPATH=$PWD /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -c "import quantagent; print(quantagent.__file__)"`.
2. Ejecutar una corrida manual supervisada: `python apps/paper_trading.py --portfolio hipotesis/m5.yaml --run-once --enable`.
3. Confirmar heartbeat `completed` en `local-data/m5.db`, revisar las órdenes y stops en la cuenta paper y correr `quantagent trading reconciliar` con las claves M5.
4. Validar posiciones y equity en Alpaca; una reconciliación M5 consulta únicamente el broker y la base configurados para ese proceso.

Los pasos son una propuesta operativa; no crean secretos, no envían órdenes durante la implementación y no instalan unidades.
