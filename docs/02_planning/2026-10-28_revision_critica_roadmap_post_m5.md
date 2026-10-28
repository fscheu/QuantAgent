# Revisión crítica del roadmap después de M5

- **Estado:** revisión para Fede (2026-10-28). No cambia `2026-10-28_roadmap_post_m5.md`; los cambios que propone se aplican allí solo si Fede los acepta.
- **Pedido de Fede:** detectar qué del plan puede ser sobreingeniería o un camino improductivo, y qué se puede traer de dos repos públicos: [`virattt/dexter`](https://github.com/virattt/dexter) y [`wshobson/financial-chat`](https://github.com/wshobson/financial-chat).
- **Cómo se hizo:** los dos repos se clonaron y se leyeron (sin ejecutar nada) en dos análisis en paralelo con un modelo más barato; la revisión del plan la hizo el PM contra el código y las decisiones vigentes.

## 1. Lo más importante, en seis puntos

1. **Las cuentas no cierran a USD 2.500, y conviene decirlo ahora.** Si una estrategia le gana a comprar y mantener por 2–3 % al año (ya sería un resultado muy bueno), con USD 2.500 son USD 50–75 al año. Sharadar con historia completa cuesta USD 499 al año: el **20 % de la cuenta**. La primera cuenta real sirve para probar el sistema con plata de verdad, no para ganar plata. Eso cambia qué vale la pena pagar y construir antes de ella.
2. **Correr la prueba de cuenta chica antes, no después.** El plan pone M-cuenta después de M-ciclos. Pero la prueba es barata y puede descartar ramas enteras: el screener de M5 con 3 acciones por sector en 11 sectores son 33 posiciones de unos USD 75. Conviene correrla sobre las carteras de M4 apenas esté, y sobre M5 **antes de decidir el pago de Sharadar** (D1 de M5, al abrir L25).
3. **La capa viva de datos que propuse es más complicada de lo necesario.** Una tabla de dividendos y splits para recalcular ajustes resuelve un problema que se evita bajando de nuevo la historia completa una vez por mes: con 12–50 ETFs tarda segundos, y Yahoo ya entrega la historia ajustada a esa fecha. Cada bajada se congela como snapshot nuevo, igual que hoy. Lo que sí hay que guardar día a día es solo lo que el proveedor **no** guarda: la foto del consenso de EPS y las decisiones de paper (que ya van a la base).
4. **El asistente con IA dentro de la web es construir un chat que ya tenemos.** Fede ya habla con Claude todos los días. En vez de un chat propio en React (streaming, contexto, permisos, auditoría), conviene un **servidor MCP de solo lectura** sobre la API de la web. Así Claude (en claude.ai, en Claude Code o en este proyecto) ve las cuentas, el embudo y los datos con las mismas herramientas, y no hay que construir ni mantener el chat. El autor de `financial-chat` hizo justamente ese camino: su proyecto vigente es un servidor MCP (`maverick-mcp`).
5. **La web puede empezar con dos pantallas, no seis.** Lo que Fede necesita para revisar es Cuenta y Embudo. Inicio sale casi gratis de esas dos. Ficha, Mercado y el asistente pueden esperar a que se usen las primeras.
6. **Más cuentas paper en Alpaca no hacen falta.** Una vez que M4 pruebe que órdenes, rechazos y dividendos andan, la incubación de 3 meses de una candidata se puede hacer en la **sombra local** (sin límite de cuentas). Alpaca paper queda para la última candidata antes del dinero real. Eso evita también el riesgo de abrir usuarios con otros mails.

## 2. Revisión por hito

### M-ciclos

- **Bien:** cadencia mensual y 5 por ciclo son conservadores; el generador aprende de causas y no de números.
- **Riesgo de camino improductivo:** con datos diarios gratis de ETFs líquidos, la probabilidad de encontrar algo que pase el Sharpe deflactado es baja, y baja más con cada ciclo (el N crece). Diez ciclos son 50 intentos más sobre los ~50 que ya hay. Propuesta: poner un **punto de corte** escrito desde ahora. Por ejemplo, si después de 6 ciclos ninguna hipótesis pasa la etapa 3, se revisa el enfoque (otra frecuencia, otros datos, otro tipo de activo) en lugar de seguir con más ciclos iguales.
- **Sobreingeniería posible:** la ficha por familia de idea y la revisión trimestral de lecciones son útiles, pero conviene empezar con una tabla a mano en el catálogo y automatizar solo si se vuelve molesta.
- **Comparación que falta:** el embudo compara contra 1/N. Para la cuenta real conviene sumar la comparación más simple de todas: **comprar y mantener SPY (o una cartera 60/40) a USD 2.500**, con los mismos costos e impuestos. Si nada le gana a eso, la cuenta real puede ser simplemente eso, y el sistema la opera.

### M-cuenta

- **Bien:** es barata y es la que más información da por hora de trabajo.
- **Cambio:** adelantarla (punto 2 de §1) y sumar la comparación contra comprar y mantener.
- **Simplificación:** con cuenta cash y sin cortos (K2), PDT no aplica de hecho a estrategias de rebalanceo; alcanza con un test que verifique que ninguna estrategia abre y cierra el mismo día.

### M-web

- **Bien:** FastAPI + React es una elección razonable y estándar.
- **Riesgo:** una web con frontend propio es el componente con más mantenimiento de todo el plan (dependencias de JavaScript, versiones, diseño en teléfono). Recortes propuestos:
  - empezar con **Cuenta y Embudo**, de solo lectura;
  - **sin login propio**: acceso solo por Tailscale (ya estaba);
  - el asistente pasa a ser un **servidor MCP** (punto 4 de §1), no un chat en la web;
  - gráficos con una biblioteca hecha (`lightweight-charts` para velas, ECharts para el resto), sin gráficos propios.
- **Lo que no conviene recortar:** el diseño de pantallas antes del código. Es barato y evita rehacer.

### M-datos

- **Bien:** Parquet inmutable para lo histórico y Postgres para operación.
- **Recortes:**
  - capa viva = **bajada completa mensual + snapshot nuevo** en vez de agregar días y recalcular ajustes (punto 3 de §1). Paper sigue leyendo precios del día de Alpaca, como hoy;
  - **DuckDB cuando llegue el universo de acciones**, no antes: para 12–50 ETFs pandas alcanza. DuckDB también lee CSV, así que la web podría leer `intentos.csv` sin migrarlo;
  - **migrar el embudo a Postgres cuando la web lo necesite**, no como trabajo aparte;
  - el catálogo de datos (un YAML por dataset y un test) sí conviene ahora: es chico y ordena todo lo que sigue.

### M-manual

- **Bien:** es el de mejor relación costo/beneficio; ya está en marcha.
- **Simplificación:** el índice de decisiones se puede **generar** desde los encabezados de los documentos `-DC-` (estado, fecha, título) igual que la referencia de comandos, en lugar de mantenerlo a mano. La tabla de criterios vigentes sí es a mano, porque es chica.

### El proceso

- Este roadmap ya tiene más de 400 líneas, y Fede pidió no seguir agrandando `PLAN-CONTINUACION.md` por lo mismo. Propuesta: `PLAN-POST-M5.md` con **una pantalla por hito** (objetivo, punto de control, decisiones tomadas, cola), y el detalle queda en este roadmap.

## 3. Qué traer de los repos

### `virattt/dexter`

Qué es: un agente de terminal en TypeScript (Bun, LangChain) para investigar empresas con datos fundamentales de un proveedor pago (financialdatasets.ai). No es un sistema de trading; el README dice "educational only". Muy popular (unas 27.700 estrellas, activo en 2026), con buena ingeniería de agente y lógica financiera delgada: el DCF es un prompt que le pide al modelo hacer cuentas. Licencia ambigua (el README dice MIT, el repo no tiene archivo de licencia), así que se traen ideas, no código.

Para traer (todas chicas, como ideas):

- **Recortar resultados grandes de herramientas** antes de devolvérselos al modelo, guardando el resto en disco (`src/utils/tool-result-storage.ts`). Sirve para el servidor MCP cuando una consulta devuelve muchas filas.
- **Permisos que niegan por defecto** (`src/permissions/`): una lista explícita de herramientas permitidas; lo que no está, no se puede. Para el servidor MCP: solo lectura, sin nada que llegue al broker.
- **Registro de cada llamada a herramienta** en un archivo (`scratchpad.ts`), para auditar qué consultó el asistente.
- **Descripciones de herramientas con "cuándo usarla y cuándo no"**: mejora la elección del modelo sin código.
- **Evaluación con un juez LLM y regla de contradicción** (`src/evals/evaluator.ts`): puntaje cero si la respuesta se contradice. Idea para revisar las hipótesis del generador contra criterios estructurales (cita una prima conocida, no trae cifras de rendimiento, es refutable).

Para **no** traer: el agente libre con búsqueda web en el camino de investigación (abre la puerta a que el generador vea resultados y narrativas de mercado, justo lo que evitamos); `bash` y escritura de archivos en el asistente; la memoria que guarda límites de riesgo y da consejos; los datos de un solo proveedor sin control de "lo que se sabía en la fecha t".

### `wshobson/financial-chat`

Qué es: una demo de 2024 (Streamlit + LangGraph + Claude 3 Opus) con herramientas de análisis técnico sobre yfinance. Sin licencia, sin tests, dependencias viejas (langgraph 0.0.55), dormida desde mediados de 2024. Los gráficos se dibujan como imagen y se suben a Imgur para que los lea el modelo; el usuario no ve un gráfico interactivo.

Para traer (ideas):

- **Errores de herramienta devueltos al modelo como resultado**, para que corrija, en vez de cortar (`app/tools/utils.py`). La API de Claude ya lo admite.
- **"Las herramientas hacen las cuentas, el modelo no"** como regla del prompt: Sharpe, P&L y veredictos salen de la API, nunca de aritmética del modelo.
- **Un set chico de herramientas por tarea** (monitor de cuentas, analista del embudo), empezando con uno solo.

Para **no** traer: el código (sin licencia, APIs que ya no existen), LangServe, el ruteo entre asistentes, la subida a Imgur, y los screeners y reglas de "setup alcista" sin validar.

**Proyecto relacionado que sí vale mirar:** [`wshobson/maverick-mcp`](https://github.com/wshobson/maverick-mcp), del mismo autor, MIT, activo (octubre 2026): un servidor MCP en Python con unas 38 herramientas de mercado sobre yfinance, sin envío de órdenes. Solo se leyó su README. Sirve como referencia de nombres y forma de las herramientas para nuestro servidor MCP, y encaja con la preferencia de Fede por herramientas hechas y gratis. No reemplaza nuestros datos: no tiene control punto en el tiempo.

## 4. Cambios propuestos al roadmap (no aplicados)

1. Escribir en el roadmap que la primera cuenta real es prueba de sistema con dinero, no fuente de ingresos, y sumar la comparación contra comprar y mantener a USD 2.500.
2. Adelantar M-cuenta: correrla sobre M4 apenas exista y sobre M5 antes de decidir el pago de Sharadar.
3. Punto de corte en M-ciclos: 6 ciclos sin ninguna hipótesis que pase la etapa 3 disparan una revisión del enfoque.
4. Incubación de candidatas en la sombra local; Alpaca paper solo para la última antes del dinero real. Se descarta abrir usuarios extra.
5. M-datos: bajada completa mensual en vez de capa viva con eventos; DuckDB y la migración del embudo a Postgres cuando se necesiten.
6. M-web: arrancar con Cuenta y Embudo; el asistente pasa a ser un servidor MCP de solo lectura sobre la API, con las ideas de §3.
7. M-manual: índice de decisiones generado desde los encabezados.
8. `PLAN-POST-M5.md` corto, una pantalla por hito.
