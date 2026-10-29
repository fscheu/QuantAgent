# Partición de QuantAgent-dpr3.1

MAN-01 abarca más de 25 archivos repartidos entre interfaz, manual, despliegue y configuración. Se divide en entregas revisables de hasta diez archivos cada una; los borrados grandes se evalúan por líneas añadidas, según la nota del ticket padre.

| Hijo | Objetivo | Aceptación binaria | Revisión |
| --- | --- | --- | --- |
| QuantAgent-dpr3.1.1 | Borrar `apps/streamlit/views/manual.py` y `tests/apps/streamlit/views/test_manual.py` | Esos dos archivos no existen | probar |
| QuantAgent-dpr3.1.2 | Borrar las demás vistas Streamlit | No existe `apps/streamlit/views/` | probar |
| QuantAgent-dpr3.1.3 | Borrar entrada, servicios, utilidades y pruebas específicas restantes | No existe `apps/streamlit/` ni quedan imports/tests específicos en pruebas vigentes | probar |
| QuantAgent-dpr3.1.4 | Quitar `Dockerfile.qa`, servicio compose, healthcheck/puerto CI, `pyproject.toml` y lockfile | No quedan configuración ni dependencia activas para Streamlit | probar |
| QuantAgent-dpr3.1.5 | Quitar referencias vigentes de `README.md`, `docs/README.md` y `docs/loop/inventario-ramas-2026-10.md` | Esos tres documentos no mencionan Streamlit ni sus rutas/puerto | leer |
| QuantAgent-dpr3.1.6 | Borrar las nueve páginas enumeradas de `docs/user-manual/` | `docs/user-manual/` no existe | leer |
| QuantAgent-dpr3.1.7 | Verificar aceptación final tras completar .1–.6 | Escaneo sin usos fuera de históricos, suite sin tests específicos borrados pasa y CI queda verde | probar |

Las referencias en documentos históricos, envelopes y `docs/loop/REVIEW-LOG.md` quedan intactas. Las nueve páginas de .1.6 son `dashboard.md`, `profile-cli.md`, `analysis-and-signals.md`, `monitoring.md`, `paper-trading-automation.md`, `strategy-configuration.md`, `index.md`, `getting-started.md` y `backtesting.md`. El hijo .1.7 depende en Beads de .1.1–.1.6 y ejecuta el escaneo global, enumera excepciones históricas en la nota de implementación y verifica la suite con los tests borrados excluidos y CI verde. La cantidad de archivos por hijo se mantiene en diez o menos; los tests de las vistas de backtesting y configuration corresponden a .1.3.
