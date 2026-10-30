# Puesta en marcha paper M5

## Objetivo y alcance

La puesta en marcha dura dos semanas sin intervención manual entre ciclos. En el tiempo 1 opera solo ETFs del universo M5, con la cuenta paper M5 y su DB aisladas. No habilita otros activos ni cambia los límites de riesgo. El inicio previsto es el lunes 2026-11-02.

El punto de entrada para los ciclos es `python -m quantagent.cli trading m5-correr --modo semanal|diario`. El procedimiento de apertura es `python -m quantagent.cli trading pre-apertura --bucket m5`; el resumen diario se consulta con `python -m quantagent.cli trading digest --bucket m5 --date YYYY-MM-DD`.

## Prerrequisitos

- **[OK Fede]** Confirmar que `~/secrets/alpaca-cartera-m5.env` existe, fue cargado y verificado contra el endpoint paper de Alpaca.
- **[OK Fede]** Confirmar una cuenta Alpaca paper separada para M5. No reutilizar la cuenta de otras carteras.
- **[OK Fede]** Revisar el universo M5, límites de riesgo y saldo paper previsto.
- **[OK Fede]** Confirmar que el clon congelado `~/paper-ops` está preparado según [el runbook paper-ops](QuantAgent-av1p.5-IM-runbook-paper-ops.md), con código y datos locales aislados.
- El directorio `local-data/` debe existir y ser escribible en el clon; la ejecución crea `local-data/m5.db` y los archivos de régimen/narrativa.

## Unidades propuestas

Estas unidades son ejemplos para `systemd --user`. Son propuestas, no están instaladas ni habilitadas. Los comandos deben ejecutarse con `WorkingDirectory` en el clon congelado `~/paper-ops/QuantAgent` y el entorno de esa instalación. Los nombres y calendario se ajustan durante el ensayo.

### Ciclo semanal

Timer semanal el viernes después del cierre de mercado (o sábado, una vez verificado el cierre de la semana):

```ini
# m5-semanal.service (propuesta)
[Service]
Type=oneshot
WorkingDirectory=%h/paper-ops/QuantAgent
ExecStart=/usr/bin/env python -m quantagent.cli trading m5-correr --modo semanal
```

```ini
# m5-semanal.timer (propuesta; viernes 22:30 UTC)
[Timer]
OnCalendar=Fri *-*-* 22:30:00 UTC
Persistent=true
```

### Ciclo diario

Timer de martes a sábado a las 08:00 UTC para procesar la última sesión cerrada:

```ini
# m5-diario.service (propuesta)
[Service]
Type=oneshot
WorkingDirectory=%h/paper-ops/QuantAgent
ExecStart=/usr/bin/env python -m quantagent.cli trading m5-correr --modo diario
```

```ini
# m5-diario.timer (propuesta)
[Timer]
OnCalendar=Tue..Sat *-*-* 08:00:00 UTC
Persistent=true
```

### Preparación pre-apertura

De lunes a viernes a las 13:30 UTC:

```ini
# m5-pre-apertura.service (propuesta)
[Service]
Type=oneshot
WorkingDirectory=%h/paper-ops/QuantAgent
ExecStart=/usr/bin/env python -m quantagent.cli trading pre-apertura --bucket m5
```

```ini
# m5-pre-apertura.timer (propuesta)
[Timer]
OnCalendar=Mon..Fri *-*-* 13:30:00 UTC
Persistent=true
```

### Digest diario

El digest requiere una fecha. La unidad propuesta puede usar un wrapper local que calcule la fecha operativa de Nueva York y llame `python -m quantagent.cli trading digest --bucket m5 --date YYYY-MM-DD`. Revisar cuidadosamente el cálculo en feriados y tras fines de semana antes de activarlo.

- **[OK Fede]** Aprobar el wrapper y su regla de fecha, o elegir una fecha explícita mantenida por el operador.
- **[OK Fede]** Instalar y habilitar los cuatro timers después de completar el ensayo; confirmar que `systemctl --user list-timers` muestra solo las unidades M5 previstas.

## Ensayo antes de activar

1. **[OK Fede]** En `~/paper-ops/QuantAgent`, confirmar branch/tag congelado, `DATABASE_URL` del entorno esperado y ausencia de datos reales o credenciales live.
2. **[OK Fede]** Verificar que las credenciales cargadas apuntan a paper y que la cuenta es la cuenta M5 dedicada.
3. **[OK Fede]** Ejecutar `python -m quantagent.cli trading m5-correr --modo semanal --as-of 2026-10-30T22:30:00Z` y revisar el régimen y el CSV generado.
4. **[OK Fede]** Ejecutar el ciclo diario con una fecha de ensayo aprobada y revisar intents, órdenes y stops en Alpaca paper.
5. **[OK Fede]** Ejecutar `python -m quantagent.cli trading pre-apertura --bucket m5` y confirmar que no hay símbolos `sin_stop`.
6. **[OK Fede]** Ejecutar digest y reconciliación; revisar DB, órdenes abiertas, stops de broker y logs antes de habilitar timers.

No cargar órdenes de ensayo sin aprobación explícita de Fede. Los comandos de ciclo pueden enviar órdenes a la cuenta paper.

## Criterios para las dos semanas

Durante dos semanas desde el 2026-11-02, revisar cada ejecución y registrar el resultado:

- Cada posición tiene stops protectores en el broker.
- Ningún rechazo queda sin explicación en el digest/log.
- La reconciliación no muestra divergencias entre posiciones locales y broker.
- Los timers completan cada ciclo esperado; cualquier código de salida 1 se investiga antes del siguiente ciclo.

La ventana solo cuenta como sin intervención si se cumplen los tres primeros criterios y no hubo acciones manuales para corregir posiciones u órdenes. Si un criterio falla, frenar y escalar a Fede; no corregir DB o sombra a mano.

## Frenado y rollback

- **[OK Fede]** Frenar M5 deshabilitando únicamente `m5-semanal.timer`, `m5-diario.timer`, `m5-pre-apertura.timer` y el timer del digest con `systemctl --user disable --now <unidad>.timer`.
- **[OK Fede]** Verificar con `systemctl --user list-timers` que no queden timers M5 activos y revisar `systemctl --user status`/journal de los servicios.
- **[OK Fede]** Preservar DB, régimen, narrativas y logs; copiar `local-data/` a un backup fechado antes de cualquier rollback.
- **[OK Fede]** Volver al tag congelado anterior en `~/paper-ops` si el código debe revertirse. No borrar órdenes, cerrar posiciones ni modificar DB manualmente; cualquier liquidación requiere una decisión de Fede.

La instalación, habilitación, cambio de versión y reanudación requieren **[OK Fede]**.
