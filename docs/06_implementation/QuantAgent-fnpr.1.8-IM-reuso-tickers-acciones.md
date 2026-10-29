# QuantAgent-fnpr.1.8 — Reúso de tickers y alias en el snapshot de acciones

Delta sobre [QuantAgent-fnpr.1.3](QuantAgent-fnpr.1.3-IM-snapshot-acciones.md). Snapshot nuevo: `acciones-sp500-2026-10b` (el `-2026-10` queda como estaba).

## Cambios

- **Cruce Yahoo/WIKI** (`_coincidencia` en `quantagent/data/acciones.py`): si un ticker está en las dos fuentes, se comparan los retornos diarios 2007..2018-03-27. Si la fracción de sesiones con diferencia < 0,5 % es menor a 0,9, la serie de Yahoo es de otra empresa (reúso del ticker) y se usa WIKI. Resultado real: BBT 0,31, CB 0,62, CPWR 0,27, DD 0,43, IR 0,30.
- **Descartados a mano** (`DESCARTADOS_YAHOO`): tickers sin WIKI cuya serie de Yahoo es de otra empresa: BUD, CCU, EP, EQ, GENZ, KG, MI, SII, TT, UST, WB. Quedan sin precio.
- **Alias**: 25 renombres en Yahoo (FB→META, ABC→COR, UTX→RTX, etc.) y AABA→YHOO en WIKI, en [`alias_tickers.csv`](../../quantagent/data/tablas/alias_tickers.csv). Todos pasan por el cruce con WIKI.
- El manifiesto guarda `reuso_cambiados_a_wiki` y `yahoo_sin_wiki` (rango y precios de cada ticker de Yahoo sin WIKI, para revisión). `write_snapshot` acepta `extra` para eso.

## Cobertura real

`verify` OK: 681 símbolos (488 Yahoo, 193 WIKI), 2.621.842 filas. Miembro-días con precio: **94,22 %**.

| 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 82,57 | 86,96 | 90,69 | 93,17 | 95,03 | 96,81 | 98,54 | 99,44 | 99,60 | 99,60 | 99,62 | 88,52 |

Saltos > 50 %: 74 en 44 tickers (antes 166). Desaparecen CPWR y EP; el resto se concentra en financieras de 2008 (FMCC, FNMA, AIG, GNW, C) y parecen reales.

## Revisión de `yahoo_sin_wiki` (no descartados)

ANET, BF.B, BRK.B, CCEP, CPRI, EVRG, FCPT, FMCC, FNMA, JEF, KDP, KEYS, LIN, LW, VIAV. En EVRG, JEF, KDP y LIN la serie anterior al cambio de nombre es la del predecesor legal (Westar, Leucadia, Dr Pepper Snapple, Praxair); solo se usan los días de pertenencia, posteriores al cambio.

## Limitaciones que siguen

- El hueco de 2007-2008 (LEH, BSC, MER, WAMU, CFC y los descartados) no tiene fuente gratis.
- 82 tickers terminan antes de salir del índice, la mayoría el 2018-03-27 por el corte de WIKI. Casos: Yahoo sin historia (AET, CELG, DFS…), Yahoo con 1-2 filas (AVB, EA, EQR) o miembros que estuvieron en el índice antes de la serie de Yahoo (DG, HLT, FOX), donde la regla de no empalmar elige WIKI.
