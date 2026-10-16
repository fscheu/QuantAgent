# QuantAgent-d4cq.14.2: partición de filtro CLI y veredicto

La estimación anterior de 95 líneas no alcanza para la implementación y
todos los tests solicitados: estimo 140 (95 de integración y 45 de bordes).
Esta entrega no cambia código. Conserva los criterios y umbrales del padre.

| Hijo | Objetivo | Líneas estimadas con tests | Revisión |
|---|---|---:|---|
| QuantAgent-d4cq.14.2.1 | Filtro por activo, tabla, intentos y prueba de integración real | 95 | probar |
| QuantAgent-d4cq.14.2.2 | Pruebas de borde del veredicto sobre doce activos | 45 | leer |

Cadena: H06.1 → .14.2.1 → .14.2.2. El padre .14.2 sigue in_progress
hasta integrar ambos hijos; H06.3 sigue esperando al padre.
La cola L9 reemplaza la fila del padre por los dos hijos.

## Aceptación binaria

- .1: snapshot sintético de dos ETFs, grilla mínima y filtro real producen
  PASA, tabla de Sharpe/trades y dos intentos etapa 1 con hipótesis.
  Un espía que llama al filtro real verifica rango 2007–2018 por activo;
  el rechazo previo sigue registrando etapa 0. FAIL si falta algún registro
  o llegan fechas fuera del rango. Se adapta el test del camino admitido.
- .2: nueve de doce bajo 0,10 producen CORTA; ocho producen PASA;
  exactamente 0,10 no cuenta como bajo. Tabla y doce intentos coinciden
  con los resultados. FAIL si mutar cualquiera de las desigualdades
  no rompe tests. Para fijar exactamente 0,10 se sustituye solo el filtro
  por resultados por activo; carga, admisión, CLI y registro siguen reales.
  La integración del filtro real queda cubierta por .1.

Archivos de ambos hijos: `quantagent/cli/hipotesis.py` y
`tests/test_cli_hipotesis.py`; el segundo modifica código solo ante un fallo.
No se cambian motores, candado, universo ni umbrales.

Tests por hijo y para revisar esta partición:
`python -m pytest -q tests/test_cli_hipotesis.py tests/test_admision.py tests/test_fabrica_filtro.py`.
Además: `scripts/loop/smoke.sh`. No se corre la suite completa.
