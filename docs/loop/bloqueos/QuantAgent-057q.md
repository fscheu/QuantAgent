# Bloqueo: QuantAgent-057q

El ticket pide un error estándar de ΔSharpe con Ledoit–Wolf o bootstrap por bloques, pero el criterio sintético exige √((2−2ρ)/T), fórmula que presupone retornos estandarizados i.i.d. y no fija si T cuenta sesiones o períodos anualizados. Esa diferencia cambia tanto el estimador como la cifra reportada.

Destrabador: definir si se implementa el estimador i.i.d. de la fórmula del criterio o bootstrap por bloques (y la unidad de T). Después se puede fijar el largo de bloque y agregarlo al informe junto al ΔSharpe.
