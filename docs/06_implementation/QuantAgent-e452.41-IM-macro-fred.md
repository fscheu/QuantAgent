# QuantAgent-e452.41: series macro de FRED

| Serie | Estado | Frecuencia | Uso |
|---|---|---|---|
| T10Y2Y | Activa | Diaria | Último valor, cambio mensual y puntos del último año |
| USREC | Persistida | Mensual | Rachas que forman `RecessionPeriod[]` |
| Otras nueve series del contrato | Próximamente | Según contrato | Sin observaciones expuestas |

Actualizar desde 2000-01-01 en la primera ingesta; las siguientes consultas comienzan siete días antes de la última observación. La clave se lee de `FRED_API_KEY` y no se imprime.

```sh
quantagent mercado macro actualizar
```

Sin calendario de publicación, T10Y2Y queda `stale` después de más de cuatro días hábiles sin observación: `ceil(1,5 × 1 día hábil)` más dos días de gracia. Si se proporciona una fecha esperada, la gracia se cuenta desde esa fecha.

Para producción, correr `quantagent api migrate-paper-db` y luego el comando de actualización.
