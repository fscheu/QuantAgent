# QuantAgent-lwji: partición de S01 (M1)

Estimación previa al código: 160–190 líneas, incluidos tests. Se divide
según PLAN-CONTINUACION.md §3.4; esta entrega no cambia ejecución ni métricas.

El informe actual solo lee estados e intentos, no curvas. La función
`_curva_y_v30_cartera` reconstruye la cartera pero también registra intentos
si recibe un snapshot: no sirve para un informe de solo lectura.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---|---|
| QuantAgent-lwji.1 | Sharpe en exceso de DGS3MO para dos curvas, con tasa compartida | Retorno igual a tasa da ≈ 0; fixture variable comparte tasa sin anticipación; falta de cobertura explícita | 70–90 líneas | leer |
| QuantAgent-lwji.2 | Integrar el cálculo en el HTML desde curvas guardadas | HTML con ambos valores; registros y DSR idénticos; sin motor ni ventanas nuevas | 80–100 líneas | probar |

El segundo depende del primero. Ambos incluyen sus tests en la estimación;
si alguno excede 100 líneas al estimar, se vuelve a partir antes de implementar.
Los detalles, archivos autorizados y comandos dirigidos están en cada hijo.
La conexión mínima del CLI al informe está incluida en el segundo hijo.

Se conserva el corte actual, el formato de `intentos.csv` y la base del DSR.
No se consulta `local-data/`. S04 y S09 siguen dependiendo del padre S01:
cerrarlo solo cuando ambos hijos estén integrados y sus criterios verificados.
