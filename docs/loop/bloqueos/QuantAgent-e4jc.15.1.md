# Bloqueo: QuantAgent-e4jc.15.1

La consulta actual no satisface la aceptación. `Position` no tiene ambiente y su lectura filtra
por símbolo, así que no demuestra aislamiento PAPER. `consultar_datos_paper` devuelve trades
cerrados, no órdenes, y expone cantidad y precio por separado, sin calcular cantidad × precio.
Además, `quantagent/models.py` no declara un modelo ni una tabla de dividendos; la revisión de
`QuantAgent-e4jc.15` ya señaló esa ausencia.

La aceptación exige que la DB SQLite sintética devuelva cantidad × precio, órdenes y dividendos
de la sesión aislados a PAPER. Resolverlo requiere definir cómo se asigna ambiente a posiciones,
qué órdenes entran en el digest y cómo se persisten y originan los dividendos. El ticket no
especifica esos contratos y no se pueden inferir de los modelos existentes.

Destraba: decidir el contrato de ambiente para posiciones, alcance de órdenes y modelo/origen
de dividendos, o ajustar la aceptación a los datos que hoy existen. No se modificó código.
