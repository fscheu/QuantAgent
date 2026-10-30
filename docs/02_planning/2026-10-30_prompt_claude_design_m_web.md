# Prompt para Claude Design: sistema de diseño y pantallas de M-web

- **Fecha:** 2026-10-30. **Tipo:** documento de trabajo, sin código ni tickets.
- **Pedido de Fede:** un prompt completo para la herramienta Design que genere un sistema de diseño y pantallas modernas para las visualizaciones de M-web.
- **Idioma del prompt:** español. Design lo entiende bien y Fede lo revisa sin traducir. El texto que aparece **dentro** de la app también va en español, con los términos de trading en inglés cuando así se usan (stop, P&L, benchmark).
- **Fuentes:** `PLAN-POST-M5.md` (M-web), `2026-10-28_roadmap_post_m5.md` §3, `2026-10-29_inventario_koyfin_para_m_web.md`, lote L34 (epic `QuantAgent-e452`, tickets WEB-01 a WEB-05) y `quantagent/models.py` para los campos reales de órdenes y posiciones.

## Dónde se usa Design

Design es un tipo de Artifact ("Design canvas": artboards vivos sobre un lienzo). Desde una sesión de este proyecto se puede crear con la herramienta de Artifacts, así que el prompt se puede pegar acá y Claude arma el lienzo en el momento. También se puede crear desde la sección de Artifacts. No es un camino o el otro: es la misma herramienta. Lo que no pude verificar es si la interfaz del proyecto le muestra a Fede un botón propio para Design; si no lo ve, que use la sección de Artifacts.

## Cómo usarlo

1. Pegar el bloque de abajo tal cual en una conversación nueva de Design.
2. Pedir primero **solo el sistema de diseño y la pantalla Cuenta** (paso 1 y 2 del bloque). Revisar. Recién después pedir Embudo, Inicio y Mercado, de a una.
3. Los datos del prompt son de ejemplo. Los campos de Cuenta salen de los tickets WEB-03 y WEB-04; los de Embudo y Mercado son los planeados, todavía sin API.
4. Esto sirve de punto de partida para las sesiones de diseño con Fede (decisión W3, todavía sin fecha). El diseño aprobado es la condición para abrir tickets de construcción del frontend.

---

## El prompt (copiar desde acá)

````markdown
# Pedido: sistema de diseño y pantallas para QuantAgent Web

Necesito que diseñes el sistema de diseño y las pantallas principales de una aplicación web de monitoreo para una plataforma de trading algorítmico. Entregá primero el sistema de diseño, después las pantallas, en versión teléfono y escritorio, con datos de ejemplo realistas.

## 1. Producto y contexto

QuantAgent es una plataforma personal de trading algorítmico en Python. Genera estrategias, las prueba con datos históricos y las opera en cuentas **paper** (simuladas, en el broker Alpaca) y, más adelante, **live** (dinero real, cuenta chica de unos USD 2.500). Un **embudo** de etapas filtra las hipótesis de estrategia hasta dejar unas pocas candidatas.

La aplicación web es de **solo lectura**: se mira, no se opera. No hay botones de comprar, vender ni cancelar, y nunca habla con el broker. Reemplaza a una interfaz anterior en Streamlit que ya no sirve. Se accede por una red privada (Tailscale), sin registro ni login visible en el diseño.

Stack previsto (para que el diseño sea construible): React + TypeScript, backend FastAPI. Los gráficos de velas irían con `lightweight-charts` de TradingView; el resto de los gráficos con una biblioteca de React a elección. Los tokens se van a traducir a variables CSS, así que entregalos con nombres claros.

## 2. Usuario

Un solo usuario principal: **Fede**, dueño del sistema. No es experto en finanzas ni en diseño. Entiende lo que ve si está bien explicado. Lee en español rioplatense. Revisa seguido **desde el teléfono**, a veces en cinco minutos, y a veces se sienta con la compu a mirar a fondo. Quiere saber rápido tres cosas: ¿está todo bien?, ¿qué cambió desde la última vez?, ¿por qué pasó esto?

Consecuencias de diseño:
- **Teléfono primero.** Tarjetas apiladas. **Nada de tablas anchas** ni scroll horizontal. En escritorio, las mismas tarjetas se reorganizan en grilla.
- **Cada término técnico se explica.** Un ícono "¿qué es esto?" junto a cada término (stop, P&L, benchmark, drawdown, sombra, etapa 3) abre una explicación breve en palabras simples. Los términos mantienen su nombre en inglés cuando así se usan.
- **Explicar el porqué.** Cuando algo falla o cae (una orden rechazada, una hipótesis descartada), el motivo se ve en la misma tarjeta, no en otra pantalla.
- Tono sobrio. Es una herramienta de trabajo, no una app de brokers para principiantes ni un casino. Sin confeti, sin gamificación, sin rojo y verde estridentes.

## 3. Sistema de diseño (entregar primero)

Definí y mostrá en una página de especificación:

**Tokens**
- Color: neutros (superficie, superficie elevada, borde, texto primario, secundario, apagado), acento de marca, y colores **semánticos** para estado: positivo, negativo, advertencia, informativo, neutro. Los de ganancia/pérdida no pueden depender solo del color: acompañar con signo (+/−), flecha o ícono (accesible para daltonismo).
- Paleta categórica de 6 a 8 colores para series de gráficos, distinguibles entre sí y legibles en claro y oscuro.
- Tipografía: una sans legible para texto y una **tabular/monoespaciada para números**, de modo que las cifras alineen. Escala de tamaños y pesos.
- Espaciado (base de 4 px), radios, sombras o bordes, elevación.
- Movimiento: transiciones sutiles y cortas; respetar "reducir movimiento".

**Modo claro y oscuro**, ambos completos, con el mismo nivel de cuidado. Cada color existe en las dos versiones.

**Densidad**: dos modos. *Cómoda* (por defecto en teléfono) y *compacta* (opción en escritorio para ver más filas). Mostrá un componente en las dos.

**Componentes** (con estados: normal, hover, foco, deshabilitado, cargando, vacío, error):
- Tarjeta de métrica (valor grande, variación del día con signo y color semántico, mini gráfico opcional).
- Tarjeta de cuenta (nombre, entorno paper/live, valor, día, semana, alarma si la hay).
- Fila-tarjeta de posición y de orden (versión apilada para teléfono; versión fila de tabla solo en escritorio).
- Insignia de estado (ejecutada, pendiente, rechazada, cancelada; paper/live; etapa del embudo).
- Barra de distancia al stop (un indicador visual de cuánto falta para que salte el stop).
- Aviso de alarma (con severidad) y aviso de dato faltante o desactualizado ("sin datos desde hace 2 h").
- Tooltip / panel "¿qué es esto?".
- Selector de rango de fechas y de entorno; filtros en forma de chips.
- Barra de navegación: inferior en teléfono, lateral en escritorio.
- Tabla densa (escritorio) con orden por columna, y su versión en tarjetas (teléfono).
- Gráficos: ver sección 5.
- Estados vacíos y de error amables, con explicación y qué hacer.

**Accesibilidad**: contraste AA como mínimo, objetivos táctiles de 44 px, foco visible, no comunicar nada solo con color.

## 4. Pantallas

Diseñá cada pantalla en **teléfono (390 px)** y **escritorio (1440 px)**, en claro y oscuro. Empezá por Cuenta; después de que revise, seguí con las demás.

### 4.1 Cuenta (la más importante; primera entrega)
Responde: *¿cómo está mi cuenta y qué hizo hoy?* Selector arriba para elegir cuenta (paper M4, paper M5, live; todas bajo la misma API).

Datos disponibles (reales, de la API en construcción):
- **Resumen**: valor total de la cuenta, variación del día (monto y %), variación de la semana (monto y %), entorno (paper/live), hora de la última actualización.
- **Posiciones**: símbolo, cantidad, precio medio de entrada, precio actual, valor, ganancia o pérdida no realizada (monto y %), **stop vigente** y **distancia al stop** (% y monto), fecha de apertura. La barra de distancia al stop es el elemento visual clave.
- **Órdenes del día** (con selector de fecha): símbolo, lado (compra/venta), cantidad, tipo (mercado/límite/stop), precio límite, precio stop, estado, precio promedio y cantidad ejecutada, hora. Para las **rechazadas**, el **motivo** en lenguaje claro (ejemplos para los datos: "poder de compra insuficiente", "no se permiten fracciones de acción", "mercado cerrado", "símbolo no operable"). Las rechazadas tienen que destacar sin gritar y quedar agrupadas arriba o filtrables.
- Caso borde de diseño: cuando el sistema todavía no guarda el motivo, la tarjeta muestra "motivo no disponible" con el ícono "¿qué es esto?". Diseñá ese estado.

Datos planeados (diseñar el lugar, marcarlos como "próximamente" si no entran en la primera entrega):
- Curva de valor de la cuenta contra su **benchmark** (SPY) y contra la **sombra local** (la simulación local que debería dar lo mismo que el broker), con selector de rango (1S, 1M, 3M, YTD, todo).
- Dividendos recibidos y retenciones.
- Exposición por sector contra SPY (barras de sobre y subponderación).
- Alarmas activas.

### 4.2 Embudo (primera entrega)
Responde: *¿cuántas hipótesis hay en cada etapa y por qué cayeron las que cayeron?*

Etapas, en orden: 0, 1, 2, 3, reserva, sombra, paper, candidata, producción. Cada hipótesis está en una etapa o fue descartada en alguna.
- Vista general: el embudo como visual (barras horizontales escalonadas o columnas conectadas, a elección tuya, pero legible en 390 px) con la cantidad en cada etapa.
- Por qué cayeron: para cada etapa, motivos de descarte agrupados (por ejemplo "Sharpe deflactado bajo el umbral", "poca cantidad de operaciones", "falló en la reserva 2023+"), con cantidad.
- Filtro por **ciclo** (un ciclo es una tanda de hipótesis propuesta y evaluada junta) y por fecha.
- Lista de hipótesis en cada etapa como tarjetas: nombre, idea en una línea, etapa actual, última métrica, fecha. Al tocar una, se abre la ficha (4.4).
- Resumen del último ciclo: cuántas entraron, cuántas pasaron cada etapa, cuántas cayeron.

### 4.3 Inicio (sale de las dos anteriores)
Responde: *¿está todo bien hoy?* Tarjeta por cuenta (valor, día, semana, alarma), resumen del último ciclo del embudo, y próximos eventos (rebalanceos, publicación de datos macro). Es lo primero que ve en el teléfono.

### 4.4 Ficha de hipótesis (fase posterior; diseño liviano)
La definición de la estrategia en palabras (viene de un YAML), su recorrido por el embudo como línea de tiempo, métricas clave (retorno anual, Sharpe, máximo drawdown, cantidad de operaciones), y enlaces a sus informes.

### 4.5 Mercado (fase posterior; ver sección 5)
Responde: *¿cómo está el mercado y qué dicen los datos macro?* Secciones:
- **Gráfico de un activo**: velas con volumen, medias móviles opcionales, rango de fechas, escala lineal o logarítmica, mínimo diario (sin velas intradía).
- **Comparar activos**: tabla con casillas; los tildados se dibujan en un gráfico de **rendimiento acumulado** (% desde el primer día del rango). Es el patrón principal: tabla de series + gráfico que se llena con lo que se tilda. En teléfono, la tabla se vuelve lista de tarjetas con casilla.
- **Sectores de EE. UU.**: los 11 ETFs sectoriales con rendimiento 1 día, 1 semana, 1 mes y en lo que va del año; versión mapa de calor (cuadrícula coloreada) y versión lista ordenable.
- **Macro EE. UU.**: curva de tasas (2 años, 10 años y el diferencial 10a−2a), inflación, desempleo, VIX; con **sombreado de recesiones** en el fondo de los gráficos y un glosario expandible por término.

## 5. Gráficos

Diseñá el estilo de cada tipo, siempre en claro y oscuro, con tooltip al tocar o pasar el mouse, leyenda clara y estados de carga y sin datos:
- **Línea** de curva de valor (cuenta contra benchmark contra sombra), con área bajo la curva muy sutil solo en la serie principal.
- **Velas** con volumen (estilo `lightweight-charts`); colores de vela alza/baja distintos a los de ganancia/pérdida de la UI si hace falta evitar confusión, y no basados solo en rojo/verde.
- **Rendimiento acumulado** multi-serie con la paleta categórica.
- **Barras horizontales** (embudo, exposición por sector, motivos de descarte).
- **Mapa de calor** de sectores, con escala divergente y valor numérico dentro de cada celda.
- **Mini gráficos (sparklines)** dentro de tarjetas de métrica.
- **Bandas de recesión** como fondo de series macro.
Reglas: ejes con pocas marcas, líneas de grilla apenas visibles, números tabulares, unidades siempre visibles, sin 3D, sin gráficos de torta salvo que justifiques por qué, y que ningún gráfico dependa solo del color para distinguir series.

## 6. Datos de ejemplo

Usá datos plausibles y coherentes entre pantallas: una cuenta paper de unos USD 2.500 a USD 100.000 según la cuenta, con 3 a 6 posiciones en acciones y ETFs conocidos de EE. UU. (por ejemplo SPY, QQQ, XLK, XLE, AAPL, MSFT), alguna ganancia y alguna pérdida, órdenes del día mezcladas con **dos o tres rechazadas** con motivos distintos, y una cuenta con alarma. Un embudo con unas 40 hipótesis en total, que se afinan a 1 o 2 candidatas. Marcá los datos de ejemplo como tales. No inventes nombres reales de personas ni de empresas de fondos.

## 7. Lo que NO tiene que tener

- Botones de operar, cancelar, depositar o conectar el broker.
- Registro, login, perfiles ni equipos. Un solo usuario.
- Noticias, chat de asistente embebido ni pantallas armables por el usuario (todo eso queda afuera por ahora).
- Copiar el aspecto, la marca ni el diseño de terminales financieros existentes. Pueden inspirar la utilidad (tabla + gráfico enlazados), no el aspecto.

## 8. Entregables, en este orden

1. Página de **sistema de diseño**: tokens (claro y oscuro), tipografía, componentes con sus estados, estilos de gráficos y reglas de accesibilidad.
2. Pantalla **Cuenta**: teléfono y escritorio, claro y oscuro, incluyendo estados vacío, cargando, error y "motivo no disponible".
3. Pantalla **Embudo**: teléfono y escritorio, claro y oscuro.
4. Pantalla **Inicio**: teléfono y escritorio.
5. Pantallas **Mercado** y **Ficha de hipótesis**: solo cuando yo lo pida.

Después de cada entregable, listá en pocas líneas las decisiones de diseño que tomaste y las dudas que tengas para que yo las resuelva, en vez de seguir de largo.
````

## Notas para quien lo revise

- **Qué es real y qué es plan.** Cuenta (resumen, posiciones con stop, órdenes con rechazadas y motivo) corresponde a WEB-03 y WEB-04 y sus campos salen de `Order`, `Position` y `ActivePosition`. Curva contra benchmark y sombra, dividendos y exposición por sector están en el roadmap pero no en L34. Embudo, Ficha y Mercado no tienen API todavía.
- **Motivo de rechazo.** WEB-04 aclara que si L26 todavía no creó `trade_intents`, la API devuelve solo lo que hay. Por eso el prompt pide diseñar el estado "motivo no disponible".
- **Alcance de Koyfin.** De `2026-10-29_inventario_koyfin_para_m_web.md` §5 el prompt toma las cuatro ideas de la lista corta (tabla con casillas más rendimiento acumulado, sectores de EE. UU., macro de EE. UU., cuenta contra benchmark por sector). Lo demás del inventario quedó afuera a propósito.
- **Tema pendiente.** Si Fede quiere una pantalla en particular antes (Mercado, por ejemplo), alcanza con cambiar el orden de la sección 8.
