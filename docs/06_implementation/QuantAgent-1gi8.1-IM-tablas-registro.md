# Tablas SQL para el registro de hipótesis

## Overview

Se agregaron tablas SQLAlchemy para hipótesis, eventos, intentos y ciclos mensuales. La migración `ab12cd34ef56` parte de `9d7c4a1b2e60` y permite revertir las cuatro tablas.

## Approach

Los modelos están en `quantagent/models.py`. Los eventos conservan `hypothesis_id` nullable y tienen índices por hipótesis y ciclo; los intentos tienen índice por hipótesis. `HypothesisAttempt` documenta el mapeo de los campos de `CAMPOS_INTENTOS`: `fecha_utc`, `estrategia`, `activo`, `snapshot`, `params`, `trades`, `pnl`, `sharpe`, `hipotesis`, `etapa` y `estado` tienen columnas propias; `desde`, `hasta`, `costos`, `tamano` y `sharpe_base` se guardan en `extra` con sus nombres CSV.

Los tests de `tests/test_registro_db_modelos.py` cubren upgrade/downgrade sobre SQLite temporal, el destino de cada campo del CSV y la estabilidad de `accounts` y `backtest_runs` al importar los modelos.

## Deviations

Ninguna.

## Known Issues

La lectura y escritura del registro SQL queda fuera de este cambio; los archivos actuales siguen siendo la fuente operativa.

## Testing

No se ejecutaron migraciones ni tests durante esta implementación. El test de migración crea una base SQLite bajo `tmp_path` y configura `DATABASE_URL` únicamente para ese archivo.
