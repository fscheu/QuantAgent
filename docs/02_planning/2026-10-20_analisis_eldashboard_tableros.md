# Análisis de eldashboard.com.ar y qué podemos construir parecido

- **Estado:** borrador para que Fede decida (2026-10-20). Investigación; no hay tickets en BEADS ni código.
- **Pedido de Fede:** analizar los tableros gratis y la lista de funciones pagas de https://eldashboard.com.ar/, compararlos con nuestros datos y proponer qué sumar, cómo construirlo y de dónde sacar los datos, para ir aprendiendo y tener visualizaciones de ese estilo.
- **Actualización (2026-10-20, pedido de Fede):** el interés está en la visualización y el análisis de **bonos**, extrapolables a EEUU; el mercado argentino importa poco. La sección 13 reordena la propuesta con ese foco y reemplaza la prioridad de las secciones 7 a 10.
- **Decisiones de Fede (2026-10-20):** biblioteca de bonos propia y chica; ampliar el snapshot de FRED; sumar `plotly`; Argentina afuera por ahora. **No entra en la cola actual**: queda para después de lo ya planificado, sin tickets en BEADS. Detalle en 13.6.
- **Cómo se leyó el sitio:** se bajaron la portada y las seis páginas públicas (soberanos, tasa fija, A3, renta variable, FX MEP, calendario). Las tablas y los textos se leyeron como texto. **No se vieron los gráficos** (se cargan con JavaScript), así que lo que dice este documento sobre gráficos sale de los títulos, los controles y la lista de funciones, no de haberlos mirado. Los precios de Pro y Mesa no están publicados.

## 1. Qué es el sitio

Un tablero del mercado argentino hecho por "Data for Traders". Cubre bonos del Estado, tasas en pesos, dólar futuro, acciones y tipo de cambio. Dice que no es un broker y que no recomienda comprar ni vender. Los precios gratis tienen 20 minutos de demora. Los tableros están firmados en X como @BenP_np.

La idea central, y lo que lo hace útil: **toma datos públicos que existen en todas partes (precios de bonos, cotizaciones) y los transforma en números que un operador necesita** (rendimiento, sensibilidad, curva, comparaciones históricas). No inventa datos; calcula bien.

## 2. Glosario para leer el tablero

Fede no es experto en trading, así que acá van los términos del sitio en palabras simples.

- **Bono soberano:** un préstamo que le hacés al Estado. Te paga cupones (intereses) y te devuelve el capital al vencimiento. Bonares: bonos argentinos con ley argentina. Globales: con ley de Nueva York. Bopreales: bonos del BCRA para importadores.
- **TIR (YTM en el sitio):** el rendimiento anual que ganás si comprás hoy a ese precio y mantenés hasta el final. Sube cuando el precio baja.
- **Duration (MD):** cuánto cambia el precio de un bono si la tasa se mueve 1 punto. Duration 5 significa que, si la tasa sube 1%, el bono pierde cerca de 5%. Mide el riesgo de tasa.
- **Convexidad (CX):** corrección fina de lo anterior. Dice que la pérdida y la ganancia no son simétricas.
- **Curva:** gráfico de rendimiento contra plazo. Si los plazos largos rinden más, la curva es "normal". Si los cortos rinden más, está "invertida" y suele avisar de problemas.
- **VWAP:** precio promedio de la rueda ponderado por volumen. Es más difícil de manipular que el último precio (Last), que puede moverlo una orden chica. El sitio lo explica en la página de acciones.
- **TNA / TEM:** tasa nominal anual / tasa efectiva mensual. Son formas de decir lo mismo con distinta unidad.
- **Instrumentos en pesos:** LECAP/BONCAP (tasa fija), CER (ajustan por inflación), TAMAR y BADLAR (acompañan una tasa de bancos), dólar linked (siguen al dólar oficial), duales (pagan lo mejor de dos opciones).
- **Dólar MEP / CCL / oficial / "canje":** distintas formas legales de comprar dólares con pesos, cada una con su precio. **Brecha:** diferencia porcentual entre el MEP o el CCL y el oficial. Es el termómetro de cuánta presión hay sobre el peso.
- **Dólar futuro (DLR):** contrato para comprar dólares a una fecha futura a un precio fijado hoy. El precio de cada contrato contra el dólar de hoy da una **tasa implícita**: lo que el mercado dice que costará el dinero en pesos.
- **Interés abierto (I.A.):** cuántos contratos siguen vivos. Volumen es cuántos se operaron hoy.
- **Forward implícito:** la tasa futura que "esconde" la curva de hoy. Ejemplo: si 1 año rinde 20% y 2 años rinde 25%, el segundo año rinde cerca de 30%.
- **Cointegración:** dos precios que se mueven separados pero tienden a volver a una relación estable. Sirve para apostar a que una distancia anormal se cierra (reversión a la media).
- **Nelson-Siegel / Diebold-Li:** fórmulas con pocos parámetros que dibujan una curva suave a través de los bonos disponibles. Sirven para ver cuál está "caro" o "barato" respecto de la curva.
- **Breakeven:** valor de equilibrio. Ejemplo: la inflación a la que da igual tener un bono CER o uno de tasa fija.
- **Riesgo país:** diferencia de rendimiento entre los bonos argentinos en dólares y los del Tesoro de EEUU. Cuánto más pide el mercado por prestarle a Argentina.

## 3. Los tableros gratis (sin cuenta)

| Tablero | Qué responde |
|---|---|
| Soberanos HD (`/soberanos`) | ¿Cuánto rinden hoy los bonos en dólares y qué tan sensibles son? |
| Curva Pesos (`/tasa-fija`) | ¿Cuánto rinde cada tipo de instrumento en pesos? |
| Futuros A3 (`/a3`) | ¿Qué tasa en pesos descuenta el mercado de dólar futuro? |
| Renta Variable (`/renta-variable`) | ¿Cómo está el panel líder de acciones (20 papeles)? |
| Series FX (`/fx/mep`) | ¿Cómo evolucionó cada dólar y su brecha? |
| Calendario (`/calendario`) | ¿Qué paga, vence o se publica y cuándo? |

Además, cada bono y cada acción tiene su ficha con el cierre histórico, y hay una superficie 3D de la curva de Bonares desde 2020.

**Soberanos HD.** Tres tablas (Bonares, Globales, Bopreales) con VWAP, variación diaria, TIR, duration y convexidad. Arriba, el panel de dólares (A3500, spot, MEP, CCL, canje y las dos brechas), una tira de variaciones del día (Merval, S&P 500, los tres grupos de bonos) y las medianas de TIR separadas por duration corta y larga. Selector de moneda: pesos, MEP, CCL. Sub-páginas: Histórico, Superficie de tasas, Ratios, Implícitas, Forwards y Calculadora con VAN y Sensibilidades.

**Curva Pesos.** Siete tablas: LECAP/BONCAP, CER (tramo corto y largo), TAMAR, BADLAR, dólar linked, duales (con el "breakeven" de devaluación o de inflación mensual) y tasas de referencia (caución, plazo fijo, TAMAR mayorista). Cada fila: ticker, vencimiento, días, precio, TNA, TIR, TEM.

**Futuros A3.** La curva de dólar futuro de 14 contratos (hoy octubre 2026 a noviembre 2027) con precio, ajuste, días, tasa directa, TNA, TEM, volumen e interés abierto, más el resumen por tramos (0-90 días ≈ 21%, 91-180 ≈ 23%, 181+ ≈ 25% anual). Un contrato sin operaciones se completa con el promedio de compra y venta ("teórico mid").

**Renta Variable.** Panel líder de 20 papeles con compra, venta, volúmenes, último, variación, mínimo, máximo, rango y hora. Filtros de plazo (24hs/CI), moneda (ARS/MEP/CCL) y precio (Last/VWAP). Pantallas extra: Múltiplos, Riesgo y un arbitraje "Roll CI/24hs". Muestra el índice Merval arriba.

**Series FX.** Ejemplo: dólar MEP implícito en el AL30. Tarjetas (último, día, 30 días, 12 meses, año a la fecha, máximo y mínimo de 1 año), selector de rango (1M a 10A y todo), conmutadores **nominal/real** (descontando inflación), **percentiles** y escala lineal/log. 3.185 observaciones desde 2013. Incluye una nota metodológica de qué bono se usa en cada época.

**Calendario.** Grilla mensual y tira semanal con licitaciones, vencimientos, pagos, publicación de datos (IPC, EMAE, REM, CPI y empleo de EEUU, FOMC, PCE) y días sin liquidación. Filtros por moneda, familia de bono y ticker. Botón para exportar a `.ics`. Los montos de cupón y amortización son **solo para Pro**.

## 4. Lo que se paga (según el sitio; sin precios publicados)

- **Pro (operadores activos):** modo trading con compra/venta/último y TNA/TIR, sensibilidades, percentiles históricos, desglose de flujos, ratios históricos entre bonos, calculadora de VAN, superficie 3D interactiva con varios modelos de ajuste de curva.
- **Mesa (todo Pro más valor relativo e implícitos):** ranking de oportunidades de reversión entre pares por cointegración, forwards implícitos de la curva en pesos y en dólares, sintéticos con futuros, carry de acciones, obligaciones negociables (ONs) y subsoberanos, licitaciones del Tesoro, volumen de renta fija, préstamo de títulos, cruces de FX y macro implícita (dólar contra banda y futuros, TAMAR implícita, inflación breakeven, tipo de cambio real, riesgo país). Una herramienta de cartera de renta fija figura "en preparación".
- **Corporativo:** por contacto, para equipos.
- **Pro y Mesa no están abiertos al público**: hay que dejar el mail.

Lectura: la parte gratis es el tablero de estado ("qué pasa hoy"). La parte paga es **el análisis encima del estado**: percentiles, comparaciones históricas, pares, implícitos, flujos. Ahí está el valor, y también lo que más se aprende construyendo.

## 5. Qué datos tenemos hoy

Verificado en el repo y en `local-data/` el 2026-10-20.

- **Precios diarios de 29 ETFs de EEUU** (`etf-1d-2026-10-amplio`, fuente Yahoo, 2007-01-03 a 2026-10-08, congelado con hashes). Cubre acciones de EEUU y por sector, mercados emergentes, bonos de distintos plazos, oro, plata, petróleo, materias primas, bienes raíces y monedas (euro).
- **Macro de EEUU en snapshot FRED/ALFRED** (`macro-2026-10`, hecho el 2026-10-17, con todas las versiones de cada dato): tasas del Tesoro a 3 meses, 2 y 10 años; diferencia 10 años menos 2 años; diferencia de crédito (Baa menos Tesoro 10 años); tasa de la Fed; inflación (CPI); desempleo; empleo (nóminas) y producción industrial. Desde 1919 a 1986 según la serie, hasta 2026-09 o 10.
- **Catálogo de primas** (`catalogo/primas.yaml`): lista de ideas con respaldo académico (tendencia, momentum, baja volatilidad, estacionalidad, carry de bonos por la pendiente de la curva, etc.), cada una con su universo de activos y su veredicto cuando ya se probó.
- **Fábrica de estrategias (M3)** y su embudo: registra todas las hipótesis, intentos y Sharpe deflactado. Es contabilidad de "qué probamos y qué aprendimos".
- **Alpaca (M4, todavía no conectado):** velas de acciones y ETFs de EEUU para paper trading. Hoy sin claves en la VM.
- **Interfaz:** Streamlit en `apps/streamlit/views/` (dashboard del scheduler, backtesting, análisis, configuración, logs, órdenes, replay, paper trading). No hay ninguna vista de mercado o de datos.

**Lo que no tenemos:** ningún dato argentino (bonos, curva en pesos, dólares MEP/CCL, futuros A3, Merval). Tampoco datos intradía ni de libro de ofertas.

## 6. Comparación directa

| El sitio | Nosotros |
|---|---|
| Curva de bonos en dólares con TIR, duration y convexidad | Tenemos la curva del Tesoro de EEUU (3M, 2Y, 10Y). Faltan los plazos intermedios. |
| Curva en pesos (7 familias) | Nada. Dato argentino. |
| Dólar futuro A3 | Nada. |
| Panel de acciones | Tenemos ETFs diarios, sin mejor oferta/demanda. |
| Dólares y brecha | Nada. |
| Calendario | Tenemos fechas de datos macro de EEUU en FRED. Falta armarlo. |
| Percentiles, nominal/real, ratios entre bonos | **Todo se puede hacer con lo que ya tenemos** (ETFs, CPI de EEUU). |
| Ranking de pares por cointegración | Tenemos la fábrica con pruebas de reversión. Falta la cointegración entre pares. |
| Carry, forwards implícitos | Los forwards se calculan con la curva del Tesoro. El carry de bonos ya está en el catálogo (L14). |

Dos conclusiones. Primero, **los datos propios alcanzan para hacer casi todos los tableros del estilo con mercado de EEUU**, que es lo que de hecho vamos a operar. Segundo, los tableros "de Argentina" necesitan datos nuevos, que existen y son gratis, pero no son parte de nuestro camino a paper trading (M4) y tienen sus trampas (sección 8).

## 7. Qué conviene sumar, en orden

La idea es que cada vista **enseñe un concepto** y que además alimente decisiones de la fábrica. Todas viven como páginas nuevas de Streamlit.

### Nivel 1: con los datos que ya tenemos (sin fuentes nuevas)

**A. Ficha de serie (como `/fx/mep`).** Para cualquier ETF o serie macro: último, variación del día/30 días/12 meses/año, máximo y mínimo de un año, selector de rango, conmutador **nominal/real** (deflactando con el CPI de FRED), **percentil histórico** y escala lineal/log. Es la pieza más reutilizable: todo el resto la usa. Enseña qué es un retorno real y por qué un percentil dice más que un número suelto.

**B. Curva del Tesoro (como Soberanos HD).** Curva de hoy (3M, 2Y, 10Y), curva de hace 1 mes/1 año superpuesta, tira con la pendiente 10Y-2Y (invertida o no) y el diferencial de crédito. Bajar más plazos de FRED (DGS1, DGS5, DGS7, DGS30, mismas fuentes que ya usamos) lleva la curva a 7-8 puntos. Enseña curva, inversión y por qué mueve a los bonos. **Aprovecha el hecho de que ALFRED guarda las revisiones**: podemos mostrar qué sabía el mercado en cada fecha, algo que el sitio no hace.

**C. Tablero de ETFs (como el panel de acciones).** Una fila por ETF: retorno 1 día, 1 mes, 12 meses, volatilidad, caída desde el máximo (drawdown), percentil de precio. Con filtro por clase (acciones, bonos, materias primas). Enseña volatilidad y drawdown.

**D. Relaciones entre activos (como Ratios y correlaciones).** Mapa de calor de correlaciones entre los 29 ETFs y ratios históricos de pares clásicos (SPY/TLT acciones contra bonos largos, GLD/SPY oro contra bolsa, XLU/SPY defensivas contra mercado, EEM/SPY emergentes). Enseña diversificación y qué cambia en las crisis.

**E. Pares con reversión (como el ranking de Mesa).** Para los pares del punto D: prueba de cointegración, distancia actual a la media en desvíos estándar y vida media de la reversión. Tabla ordenada de pares "más estirados". **Ojo:** esto es cercano a lo que hace la fábrica; conviene que sea una hipótesis más del catálogo con su veredicto, no una señal suelta. Si el par se muestra pero no pasa por el embudo, queda marcado como "informativo, no validado".

**F. Calendario macro y de la fábrica.** Fechas de publicación de empleo, CPI y FOMC (FRED publica calendario de releases) y, a la par, nuestras fechas: corridas del loop, aperturas de reserva, hitos. Es una tira semanal con filtros, sin montos.

### Nivel 2: para cuando se conecte Alpaca (M4)

**G. Panel de paper trading en vivo:** posiciones, P&L, órdenes contra las señales, con el mismo formato de tabla del sitio (último, variación, rango, hora). Ya hay `paper_trading.py`; es darle el formato del panel líder.

**H. Curva de volatilidad y sectores del día:** rotación sectorial (los 9 sectores XL*) con mapa de calor. Necesita velas más frescas que Yahoo diario.

### Nivel 3: mercado argentino (datos nuevos)

Solo vale la pena si Fede quiere aprender ese mercado o si el LLM/hipótesis va a mirar Argentina. No hace falta para M4.

**I. Dólar y brecha:** dólar oficial, MEP, CCL, brecha. Series públicas del BCRA y de sitios agregadores (ver sección 8).
**J. Curva de pesos (LECAP/BONCAP y CER):** requiere precios de bonos por cierre. Las trampas están en el cálculo (convención de días, feriados).
**K. Riesgo país y bonos en dólares:** precio de los Globales (Yahoo o brokers) contra el Tesoro de EEUU de nuestro snapshot. Es la pieza más barata de Argentina porque ya tenemos la otra mitad.
**L. Dólar futuro y acciones locales:** la parte más cara en datos (A3/BYMA, dato de pago o con demora). No recomendado.

## 8. De dónde sacar los datos

Para todo lo de Nivel 1 y 2 las fuentes **ya existen en el repo**: Yahoo (snapshots congelados), FRED/ALFRED (`quantagent/data/sources/fred.py`) y Alpaca (`quantagent/data/sources/alpaca.py`).

Para Nivel 3, candidatas (a verificar antes de elegir; algunas no las pude confirmar):

- **BCRA, API pública** (`bcra.gob.ar/apis-banco-central`): hay manuales oficiales para estadísticas cambiarias v1 (cotizaciones de divisas) y monetarias v3 (variables principales, incluye A3500 y tasas). Sin clave. La búsqueda mostró también una v4.0 mencionada por un envoltorio no oficial; hay que confirmar la versión vigente en la página oficial.
- **INDEC / datos.gob.ar:** series de tiempo (inflación, EMAE). No verificado en esta búsqueda.
- **ArgentinaDatos** (`argentinadatos.com`): API gratuita con dólares, inflación y otros. La confirmó un paquete de CRAN; no confirmé el detalle de endpoints ni los términos de uso.
- **data912 y similares:** nombran cotizaciones de bonos y CEDEARs gratis. **No pude verificar que existan hoy ni sus condiciones.** No usar sin leer los términos.
- **Yahoo:** tickers de ADRs (GGAL, YPF, BMA, PAM, etc.) y `^MERV`; Globales con símbolos de bolsa de EEUU o de Buenos Aires. Cobertura irregular.
- **BYMA / A3 / MAE (fuentes de origen):** el dato en tiempo real es de pago. No es algo que podamos esperar gratis, y no verifiqué sus términos de uso.
- **Prospectos de los bonos** (cupones, amortización, convención de días): es lo que usa el sitio para calcular TIR bien. Hay que cargarlos a mano o bajarlos de la Secretaría de Finanzas; no se extraen limpios de ningún API que yo conozca.

Regla que ya aplicamos en `QuantAgent-iuf` y `QuantAgent-7nle` (los términos de Yahoo prohíben la extracción automática): cualquier fuente nueva tiene que pasar primero por una nota de términos y de calidad, igual que `iuf` y `7nle`. Nada de scraping del propio eldashboard.com.ar: su fuente real es BYMA/A3, y copiar su tablero sería usar su trabajo sin permiso.

## 9. Cómo lo construiríamos

**Pila:** Streamlit (ya está en el proyecto). Para gráficos interactivos haría falta sumar Plotly como dependencia (hoy no está en `pyproject.toml`), y para la prueba de cointegración del punto E, `statsmodels` (tampoco está). No hace falta un front nuevo.

**Capa de datos:** una función por vista que lee **solo de snapshots congelados** (`local-data/`), igual que el resto del sistema. Nada baja datos desde la interfaz. Para frescura se hace un "snapshot de mercado" nuevo cada tanto, con nombre y fecha, como los de hoy. Así cada gráfico es reproducible y la fábrica no se contamina.

**Estructura mínima:**
- `quantagent/viz/` con funciones puras (`ficha_serie`, `curva_tesoro`, `correlaciones`, `ratios`, `pares_cointegrados`), sin Streamlit adentro y con tests contra números calculados a mano.
- `apps/streamlit/views/mercado.py` con las vistas A-F como pestañas.
- Reutilizar `snapshot.load_*` y `macro.py` ya existentes.

**Orden sugerido (cada paso es un entregable de tamaño de ticket):** A (ficha de serie) → B (curva del Tesoro) → C (tablero de ETFs) → D (correlaciones y ratios) → F (calendario) → E (pares, dentro de la fábrica). Después, si Fede quiere, K (riesgo país), que es el primer paso a Argentina con costo bajo.

**Qué no hacer:**
- No tratar estas vistas como señales. Un percentil extremo no es una estrategia; cualquier idea derivada pasa por el embudo y se cuenta como intento (si no, rompe el Sharpe deflactado).
- No agregar datos intradía hasta M4.
- No construir Nivel 3 completo "por las dudas".

## 10. Recomendación

Hacer **A, B y C** primero. Son baratas, usan datos que ya tenemos, y enseñan los tres conceptos que más se repiten en el sitio (percentil histórico, curva de tasas, riesgo por activo). Después decidir con Fede si le interesa Argentina (K) o si prefiere seguir con pares (E), que conecta directo con la fábrica.

## 11. Decisiones que le tocan a Fede

1. ¿Arrancamos por Nivel 1 (EEUU, sin datos nuevos) o querés ya algo de Argentina?
2. ¿Va en el mismo Streamlit del proyecto o en una app aparte (para no mezclar con el scheduler)? Recomendado: misma app, pestaña "Mercado".
3. ¿Las vistas pueden usar datos más frescos que el snapshot congelado (por ejemplo, un snapshot semanal) o se quedan en el corte fijo? Recomendado: snapshot semanal con nombre y fecha.
4. ¿Se crean los tickets de A-C como lote nuevo o se esperan al cierre del lote actual?

## 12. Lo que queda sin verificar

- Los gráficos del sitio y el contenido tras el login.
- Los precios de Pro y Mesa (no están publicados).
- Las condiciones y los endpoints de ArgentinaDatos, data912 y las APIs del BCRA (versión vigente).
- Cualquier término de licencia de los datos de BYMA/A3.

## 13. Foco en bonos: lo que se puede copiar del sitio, versión EEUU

**Idea central.** Casi todo lo que hace el sitio con bonos argentinos es matemática de bonos, no de Argentina: flujos de fondos, rendimiento (TIR), duration, convexidad, curva, forwards, ajuste de curva. Si construimos esa matemática una vez como biblioteca, la curva de EEUU entra hoy y la de Argentina (o cualquier otro país) entra después cambiando solo la fuente de datos y los flujos.

### 13.1 Equivalencias, tablero por tablero

| Sitio (Argentina) | Versión EEUU |
|---|---|
| Tabla de Bonares/Globales con TIR, duration, convexidad | Tabla de la curva del Tesoro por plazo (1M a 30A): rendimiento, duration y convexidad de un bono "a la par" de ese plazo, más los ETF de bonos (SHY, IEF, TLT, AGG, LQD, TIP) como referencia con precio real |
| Superficie 3D de tasas desde 2020 | Plazo por fecha por rendimiento, desde 1962 en los plazos largos. Más historia que el sitio |
| Histórico y percentiles | Percentil del rendimiento de cada plazo contra su propia historia |
| Ratios entre bonos | Pendientes y "mariposas": 2A-10A, 5A-30A, 2A-5A-10A. Diferencial de crédito (Baa menos Tesoro 10A) |
| Forwards implícitos | Tasas forward de la curva (1A dentro de 1A, 2A dentro de 3A, etc.), calculadas de la curva |
| Breakeven de inflación (CER contra tasa fija) | Inflación breakeven (TIPS contra nominal) y rendimiento real |
| Ajuste Nelson-Siegel/Diebold-Li | Lo mismo sobre la curva del Tesoro; los tres parámetros (nivel, pendiente, curvatura) son una serie de tiempo que se grafica |
| Calculadora de VAN y sensibilidades | Calculadora de bono: precio, TIR, DV01 y escenarios de +-50/100 puntos básicos |
| Calendario de licitaciones | Calendario de licitaciones del Tesoro y publicaciones macro |
| Ranking de pares por cointegración | Reversión de pendientes y mariposas (por ejemplo, si 2A-10A está estirada contra su historia) |

Glosario nuevo: **DV01** es cuánto pierde el bono en dólares si la tasa sube un punto básico (0,01%). **Mariposa** compara el plazo medio contra el promedio de los extremos: dice si la curva está más "panzona" o más plana de lo normal. **Rendimiento real** es el que queda descontando inflación; en EEUU lo dan los TIPS. **Punto básico** es 0,01%.

### 13.2 Datos de EEUU (todo gratis)

Ya tenemos en `macro-2026-10`: DGS3MO, DGS2, DGS10, T10Y2Y, BAA10Y, FEDFUNDS y CPI. Faltan, de la misma fuente FRED y con el mismo cliente (`fred.py`):

- Plazos intermedios del Tesoro: DGS1MO, DGS6MO, DGS1, DGS3, DGS5, DGS7, DGS20, DGS30.
- TIPS y breakeven: DFII5, DFII10 (rendimiento real) y T5YIE, T10YIE (inflación breakeven).
- Crédito: BAMLC0A0CM y BAMLH0A0HYM2 (grado de inversión y alto rendimiento).

Los nombres de las series salen de mi conocimiento de FRED, no los verifiqué en esta sesión; hay que confirmarlos al armar el ticket. Alternativa oficial: la curva diaria de rendimientos que publica el Tesoro de EEUU (home.treasury.gov), sin clave. El cliente de FRED ya baja las versiones revisadas (ALFRED), que es una ventaja sobre el sitio.

**Un límite importante:** FRED da **rendimientos por plazo**, no el precio de cada bono. El sitio calcula todo desde el precio de cada bono. Con FRED, duration y convexidad se calculan para un bono teórico a la par de cada plazo, que es suficiente para aprender y comparar. Para precios de bonos reales, hay tres caminos: los ETF de bonos (ya los tenemos), los resultados de licitaciones del Tesoro (API de TreasuryDirect / Fiscal Data, no verificado) o Alpaca más adelante (no sé si cubre bonos individuales; no lo verifiqué).

### 13.3 Cómo se construye

1. **`quantagent/bonos/`**, con funciones puras y tests contra números calculados a mano:
   - `flujos` y `precio_tir`: cronograma de cupones, precio a partir de la TIR y TIR a partir del precio.
   - `riesgo`: duration modificada, convexidad y DV01.
   - `curva`: interpolación, tasas forward y ajuste Nelson-Siegel (con `scipy`, que ya es dependencia).
   - `analisis`: percentiles, pendientes, mariposas y spreads contra su historia.
2. **`apps/streamlit/views/bonos.py`**, con pestañas: Curva, Histórico y percentiles, Superficie, Pendientes y mariposas, Forwards y breakeven, Calculadora.
3. **Fuente separada de la matemática.** La biblioteca no sabe de Argentina ni de EEUU: recibe flujos y precios. Argentina, si Fede la quiere, es un adaptador nuevo (bonos con cupones y calendario propios) y el riesgo país sale gratis de restar la curva del Tesoro.
4. **Todo desde snapshots congelados**, como el resto del sistema.

### 13.4 Orden propuesto

1. Biblioteca de bonos (`flujos`, `precio_tir`, `riesgo`) con tests contra ejemplos de libro. Es la base de todo y la parte que más enseña.
2. Snapshot FRED ampliado (plazos, TIPS, crédito).
3. Vista de curva con tabla de TIR, duration y convexidad, y curva de hoy contra hace 1 mes y 1 año.
4. Histórico, percentiles y superficie 3D.
5. Pendientes, mariposas y forwards.
6. Breakeven y rendimiento real (TIPS).
7. Ajuste Nelson-Siegel y serie de nivel, pendiente y curvatura.
8. Calculadora de bono con DV01 y escenarios.

Conexión con la fábrica: el carry de bonos por la pendiente de la curva ya está en el catálogo de primas (L14). Las vistas 3 a 5 son la herramienta para entender esa hipótesis antes de testearla; si de ahí sale una regla de trading, entra por el embudo y cuenta como intento.

### 13.5 Lo que cambia respecto de las secciones 7 a 11

Las vistas de ETFs, correlaciones y pares (secciones 7.C, 7.D y 7.E) pasan a segundo plano. Argentina (7, Nivel 3) queda como extensión opcional del paso 3 de 13.3. La recomendación pasa a ser: empezar por el paso 1 de 13.4. Las decisiones abiertas son ahora:

1. ¿Biblioteca de bonos propia o apoyarse en una existente (por ejemplo `QuantLib` o `scipy`)? Recomendado: propia y chica para duration, convexidad y precio, porque es lo que se quiere aprender, y `scipy` para el ajuste de curva.
2. ¿Ampliamos el snapshot de FRED con los plazos, TIPS y crédito de 13.2?
3. ¿Aprobás sumar `plotly` como dependencia (`scipy` ya está)?
4. ¿Argentina queda afuera por ahora?

### 13.6 Decisiones registradas (2026-10-20)

1. **Biblioteca de bonos:** propia y chica (precio, TIR, duration, convexidad, DV01). `scipy` para el ajuste de curva.
2. **Snapshot de FRED:** se amplía con plazos intermedios, TIPS y crédito (13.2).
3. **Dependencias:** se suma `plotly`. `scipy` ya está.
4. **Argentina:** afuera por ahora.

**Sin fecha ni cola.** Esto se retoma después de lo que está planificado hoy. No se crean tickets hasta entonces. Al retomar, el punto de partida es el paso 1 de 13.4.
