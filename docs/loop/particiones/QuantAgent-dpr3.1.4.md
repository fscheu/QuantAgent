# Partición de QuantAgent-dpr3.1.4

El cambio supera el objetivo de 100 líneas: elimina dos artefactos Docker y su dependencia, además del job de
despliegue QA completo del workflow. El repositorio no contiene lockfile de dependencias.

Se divide en entregas independientes:

| Hijo | Objetivo | Criterio binario | Revisión |
|---|---|---|---|
| `QuantAgent-dpr3.1.4.1` | Quitar `Dockerfile.qa`, `docker-compose.qa.yml` y Streamlit de `pyproject.toml` | Los dos archivos no existen y la dependencia no está declarada | probar |
| `QuantAgent-dpr3.1.4.2` | Quitar el job de despliegue QA y el healthcheck/puerto Streamlit del workflow | No queda job activo `deploy-qa` ni referencia activa Streamlit/8501 | probar |

No se encontró lockfile en el repositorio.
