# Inventario de componentes

Referencia visual: canvas, página **Sistema de diseño**, hojas 03 (Componentes A), 04 (Componentes B) y 05 (Gráficos), y página **v0.2 · Ajustes** (posiciones cortas, estados de orden, cuenta sin datos, DataTable). Si una hoja de v0.1 contradice a v0.2, manda v0.2.
Todos los componentes usan solo variables `--qa-*` de `tokens.css`. Ningún color literal en componentes.

**Estados obligatorios** para todo componente que muestra datos: `normal · cargando · vacío · error · desactualizado`.
Para controles: `normal · hover · foco · seleccionado · deshabilitado`.

Convención: `?` = prop opcional. Los tipos de datos vienen de `api-types.ts`.

---

## 1. Primitivas

### `Card`
Contenedor base. `--qa-surface`, borde `--qa-border`, radio `--qa-radius-lg`, padding `--qa-card-pad`.
| Prop | Tipo | Nota |
|---|---|---|
| `as?` | `'div' \| 'section' \| 'article' \| 'a'` | `a` cuando toda la tarjeta navega |
| `href?` | `string` | Activa hover (borde fuerte + sombra 1) y foco |
| `variant?` | `'default' \| 'dashed'` | `dashed` = próximamente/deshabilitado |

### `Num`
Envuelve cualquier cifra: `font-family: mono`, `tabular-nums`. **Toda cifra pasa por acá.**
| Prop | Tipo |
|---|---|
| `size?` | `'xl' \| 'l' \| 'm' \| 's'` (tokens `--qa-num-*`) |

### `Delta`
Variación con signo + flecha + color. Nunca solo color.
| Prop | Tipo | Nota |
|---|---|---|
| `value` | `number` | Monto |
| `pct?` | `number` | |
| `layout?` | `'inline' \| 'stacked'` | inline: `▲ +1.240,50 · +1,28 %` |
| `size?` | `'m' \| 's'` | |
- `> 0` → `--qa-pos`, flecha arriba, `+`. `< 0` → `--qa-neg`, flecha abajo, `−`. `0` → `--qa-neutral`, raya.
- `aria-label` completo: "subió 1.240,50 dólares, 1,28 por ciento".

### `Badge`
Pastilla 24 px (20 px en contextos compactos), radio pill (etapas: radio sm).
| Variante | Fondo / texto | Ícono |
|---|---|---|
| `filled` (orden ejecutada) | neutral-soft / text | tilde |
| `pending` | info-soft / info | reloj |
| `partial` | info-soft / info | círculo medio lleno |
| `rejected` | neg-soft / neg | cruz |
| `cancelled` | transparente + borde **continuo** control / text-2 | círculo tachado, "Cancelada" |
| `expired` | transparente + borde **punteado** control / text-2 | reloj de arena, "Vencida" |
| `paper` | borde punteado control / text-2 | — |
| `live` | text / surface (invertido) | punto, texto "Live · dinero real" |
| `no-data` | borde punteado control / text-2 | texto "sin datos" (junto al nombre de la cuenta) |
| `long` (lado) | borde control / text-2, radio sm, 22 px | tendencia arriba, "Larga" |
| `short` (lado) | neutral-soft + borde text-2 / text, radio sm, 22 px | tendencia abajo, "Corta" |

Notas v0.2: el estado de orden del backend se escribe `cancelled` (dos L). Cancelada y Vencida son neutras (no son errores) y se distinguen por ícono, palabra y tipo de borde. El lado de una posición es neutro a propósito: ni Larga ni Corta usan colores semánticos.
| `stage` | neutral-soft / text; `candidate` accent-soft / accent; `production` accent / on-accent | — |
| `discarded` | borde control / text-2 | cruz, "Descartada en E1" |
| `soon` | info-soft / info | texto "Próximamente" |

### `HelpTip`
Botón "?" (ícono 16–20 px, **área táctil 44 px** con `padding` + `margin` negativo).
| Prop | Tipo | Nota |
|---|---|---|
| `term` | clave de `help-texts.es.json` | |
| `vars?` | `Record<string,string>` | Rellena `{placeholders}` del ejemplo |
- Escritorio: popover (`--qa-surface-raised`, `--qa-shadow-2`, máx. 280 px). Abre con clic o Enter, cierra con Esc o clic afuera.
- Teléfono (< 640 px): hoja inferior con manija, botón cerrar de 44 px, se cierra deslizando.
- `aria-label="Qué es {title}"`, `aria-expanded`. Foco vuelve al botón al cerrar.

### `Notice`
Aviso con severidad. Ícono con forma propia por severidad.
| `severity` | Fondo | Ícono |
|---|---|---|
| `critical` | neg-soft | octógono (`role="alert"`) |
| `warning` | warn-soft | triángulo |
| `info` | info-soft | círculo "i" |
| `stale` | transparente, borde punteado warn | reloj |
Props: `title`, `message`, `since?` (mostrar "desde 14:05 ET"), `action?` (`{label, href}`).

### `Skeleton`
Bloques `--qa-surface-sunken`, radio sm. Con reducir movimiento: estático (sin brillo). Los **títulos reales** se muestran siempre; solo los datos esperan. Pasados 10 s, texto "Sigue cargando, la API está lenta".

### `EmptyState` / `ErrorState`
Ícono 28 px + título + explicación + acción opcional (botón texto acento, 44 px).
Textos de la hoja 04: "No hubo órdenes este día", "Ninguna orden rechazada hoy" (filtro), "No pudimos conectar con la API" (con pasos: Tailscale / servidor), "Sin posiciones abiertas".

### `ComingSoon`
Envuelve una sección planeada: agrega `Badge soon` en el encabezado. Puede renderizar datos de ejemplo (diseño) o un fantasma punteado (producción sin datos). Decidir por feature flag.

---

## 2. Controles

### `SegmentedControl`
`role="radiogroup"`; cada opción `role="radio"`, `aria-checked`. Fondo pista sunken, opción elegida = surface + sombra 1 + peso 600. Alto `--qa-control-h` (en teléfono mín. 40 + área 44).
Usos: rango (1S 1M 3M YTD Todo), período (30 días, 90 días, Todo), entorno (Todas, Paper, Live), escala (Lineal/Log), densidad, tema.

### `ChipGroup` / `Chip`
Filtros con contador. `aria-pressed`. Seleccionado: accent-soft + borde accent + tilde. Contador 0 → deshabilitado. Envolver en filas (`flex-wrap`), **nunca scroll horizontal**.

### `AccountSelect`
Botón con nombre + `Badge paper|live` + chevrón → listbox (`role="listbox"`) con valor de cada cuenta y alarma si la hay.
Cuenta con `status: 'no_data'`: en el botón, `Badge no-data` en lugar del de entorno; en la lista, "Sin datos todavía" con círculo punteado en lugar del monto. Se puede elegir igual. Nunca "USD 0,00".

### `DateStepper`
`‹ Hoy · jue 29/10 ›`. Flechas de 44 px; "siguiente" deshabilitada en hoy. Tocar la fecha abre un calendario.

### `DataTable` (v0.2)
Tabla genérica para Mercado, Embudo, Posiciones y Órdenes. Referencia: canvas, página "v0.2 · Ajustes", tablero 4.

```ts
type Align = 'left' | 'right';
type CardRole = 'title' | 'subtitle' | 'primary' | 'secondary' | 'hidden';

interface Column<T> {
  key: string;
  header: string;
  /** 'num' | 'money' | 'pct' | 'pp' → alineación derecha + <Num>; 'pct' agrega signo, flecha y color. */
  type: 'text' | 'num' | 'money' | 'pct' | 'pp' | 'date' | 'badge' | 'custom';
  align?: Align;                          // por defecto: derecha para numéricos, izquierda para el resto
  width?: string;                         // CSS de grid: '88px', 'minmax(150px, 1fr)'
  sortable?: boolean;
  sortValue?: (row: T) => number | string; // si no, el valor crudo
  defaultSortDir?: 'asc' | 'desc';         // numéricos: 'desc' en el primer clic
  help?: string;                           // clave de help-texts → HelpTip en el encabezado
  render?: (row: T) => ReactNode;          // para 'custom' (StopDistanceBar, Badge, etc.)
  card?: CardRole;                         // cómo aparece en la versión tarjeta
}

interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  caption: string;                         // accesible; visible u oculto
  sort?: { key: string; dir: 'asc' | 'desc' } | null;
  onSortChange?: (s: { key: string; dir: 'asc' | 'desc' } | null) => void;
  selectable?: boolean;
  selected?: Set<string>;
  onSelectionChange?: (next: Set<string>) => void;
  maxSelected?: number;                    // ej. tableConfig.maxSelectedForChart
  selectionSwatch?: (row: T, index: number) => { color: string; dash: string } | null; // muestra de serie
  subRow?: (row: T) => ReactNode | null;   // motivo de rechazo, vencida, etc.
  groupBy?: (row: T) => { id: string; label: string; tone?: 'neg' | 'neutral' } ; // "Rechazadas · 3"
  density?: 'comfortable' | 'compact';     // por defecto lee data-density
  state: 'ready' | 'loading' | 'empty' | 'error' | 'stale';
  staleAsOf?: string;
  emptyTitle?: string; emptyMessage?: string; emptyAction?: { label: string; onClick: () => void };
  errorMessage?: string; onRetry?: () => void;
  onRowClick?: (row: T) => void;           // si existe, la fila es navegable (enter/clic)
  cardBreakpoint?: number;                 // tableConfig.cardBreakpoint
}
```

Comportamiento:
- **Orden**: el encabezado ordenable es un `<button>` dentro de `role="columnheader"` con `aria-sort` (`ascending | descending | none`). Ícono: dos flechas tenues sin orden; una flecha llena en la columna activa. Primer clic: `defaultSortDir`; segundo: invierte; tercero: vuelve al orden por defecto de la tabla.
- **Numéricos**: alineados a la derecha, `<Num>`, tabular. `pct`: "▲ +3,85" / "▼ −4,60" con color semántico; sin "%" en la celda (la unidad va en el encabezado o en el caption).
- **Casillas**: columna de 44 px (32 px en compacta) con `<input type="checkbox">` real y `aria-label` ("Graficar XLK"). En el encabezado, "tildar todos" con estado indeterminado. Al llegar a `maxSelected`, las demás casillas se deshabilitan y aparece "Máximo 5: destildá una para sumar otra". Contador visible: "3 de 5 tildados".
- **Muestra de serie**: si `selectionSwatch` existe, junto al título va una línea de 18 px con el color y el trazo con que se dibuja en el gráfico.
- **Fila**: hover = sunken; foco (teclado) = contorno 2 px de foco por dentro; flechas arriba/abajo recorren filas, espacio tilda, enter abre si hay `onRowClick`.
- **Densidad**: cómoda (filas 52–56 px, texto 15, padding 16) y compacta (filas 36 px, texto 13, padding 12, casilla 32 px). Solo escritorio permite compacta.
- **Estados**: `loading` = encabezado real + `skeletonRows` filas esqueleto; `empty` = mensaje + acción dentro del cuerpo; `error` = mensaje en neg con octógono + "Reintentar ahora" + reintento automático; `stale` = franja punteada warn arriba ("Sin datos nuevos desde hace 2 h · precios de las 13:05 ET") y cifras en text-2 sin color semántico.
- **Tarjetas (< `cardBreakpoint`)**: cada fila es una tarjeta. `title` (+ `subtitle`) arriba a la izquierda, `primary` arriba a la derecha, `secondary` en grilla de 4 columnas con etiqueta chica arriba. `hidden` no aparece. Casilla de 44 px a la izquierda. El orden pasa a un selector "Ordenar por: {columna}" + botón de dirección ("Mayor" / "Menor"); la columna activa se resalta en la tarjeta con su flecha. Nunca scroll horizontal.
- **Grupos y sub-filas**: `groupBy` agrega encabezados de grupo (ej. "Rechazadas · 3" en neg-soft); `subRow` agrega una línea debajo de la fila (motivo), sin romper la grilla.

### `BottomNav` (< 1024 px) / `SideNav` (≥ 1024 px)
5 destinos (v0.3.1): Inicio, Cuenta, Backtests, Embudo, Mercado. Backtests es sección propia (ícono de reloj con flecha). En la barra inferior, etiquetas de 11,5 px y ancho flexible para que entren los 5 en 390 px. Ícono + texto siempre. Activo: `aria-current="page"`, accent-soft. Destinos aún no construidos: texto apagado + "pronto". SideNav: ícono de advertencia junto a "Cuenta" si hay alarmas. Densidad y tema al pie (escritorio) o en el ícono de ajustes de Inicio (teléfono).

---

## 3. Dominio

### `MetricCard`
Etiqueta + `Num` + `Delta` + `Sparkline?` + pie (hora). Estados de la hoja 03 (incluye "próximamente" y "desactualizado").

### `AccountCard` (Inicio)
`AccountSummary` → nombre, badge entorno, valor, hoy, semana, sparkline 30 días, alarma (`Notice` compacto) o "Sin alarmas · {meta}". Toda la tarjeta es enlace a Cuenta.
Variante `no_data`: `Card variant="dashed"`, más baja; nombre + `Badge no-data`, "Sin datos todavía" (text-2) y el motivo en una línea (text-3). Sin monto, sin variaciones, sin sparkline, sin "Sin alarmas". No cuenta para el `HomeStatusBanner`.

### `AccountSummaryCard` (Cuenta)
Valor (`num-xl`), hoy y semana (`Delta stacked`), pie: largo / corto / efectivo / órdenes de hoy. "Corto" solo aparece si `short_value > 0`.

### `AccountNoData` (Cuenta, v0.2)
Reemplaza **todo** el contenido de Cuenta cuando `status: 'no_data'`.
- Ícono de cuenta punteado, título "La cuenta {nombre} todavía no tiene datos".
- Explicación según `no_data_reason`:
  - `not_opened`: "La cuenta todavía no se abrió en el broker."
  - `not_funded`: "La cuenta existe pero todavía no tiene fondos."
  - `unknown`: "Está en la configuración, pero el broker todavía no informó nada: la cuenta no se abrió o no tiene fondos."
  - Siempre se cierra con "No es un error."
- Qué va a aparecer: "Cuando llegue el primer dato vas a ver acá el valor, las posiciones y las órdenes."
- Bloque sunken: "Última consulta al broker: {checked_at} ART · Próxima: en {n} min".
- "Mientras tanto:" + enlaces a las otras cuentas con datos.
- **No se muestra**: resumen con ceros, secciones vacías de posiciones u órdenes, gráficos, bloques "Próximamente", alarmas.
- Distinto del estado "Vacío": una cuenta vacía tiene datos (efectivo, sin posiciones) y muestra su resumen real.

### `StopDistanceBar`
| Prop | Tipo |
|---|---|
| `side` | `'long' \| 'short'` |
| `price`, `stop`, `qty` | `number`, `number \| null`, `number` (positiva) |
| `variant?` | `'compact' \| 'anatomy'` |
- Cálculo en `config.ts → stopDistance(side, price, stop, qty)`. Escala fija 0–`scalePct`.
- **El eje horizontal es el precio** (izquierda = menor, derecha = mayor). Larga: stop a la izquierda, relleno crece desde la izquierda (`anchor: 'left'`). Corta: stop a la derecha, relleno crece desde la derecha (`anchor: 'right'`). En los dos casos el relleno va del stop al precio y mide el margen.
- Distancia: larga `(precio − stop) / precio`; corta `(stop − precio) / precio`. Zonas iguales en los dos lados.
- Zona `ok`: relleno text-2. `attention`: warn + triángulo + "Atención". `near`: neg + octógono + "Cerca". `breached` (precio ya pasó el stop: salto de apertura o recompra pendiente): pista con borde neg + octógono + "Stop superado". `none`: pista punteada + "Sin stop vigente".
- Texto siempre presente: "Cerca · 1,5 % · USD 125". En tarjeta, la etiqueta del stop dice "Stop abajo · 520,00" o "Stop arriba · 81,50".
- `role="meter"` con `aria-valuenow` y `aria-valuetext` ("a 6 por ciento del stop, stop por encima del precio").
- Implementación: contenedor `display:flex; justify-content: flex-start | flex-end` según `anchor`, relleno con `width = fill`.

### `PositionCard` (teléfono) / `PositionRow` (≥ 1024 px)
Card: símbolo + **`Badge long|short`** + descripción · valor / `qty × precio · entr.` · P&L (`Delta`) / (solo cortas) línea "Gana si el precio baja: bajó de X a Y." o "Pierde si el precio sube: subió de X a Y." / `StopDistanceBar compact`.
Row: columnas Símbolo (+ descripción) · **Lado** · Cant. · Entrada · Actual · Valor · P&L no realizado · Stop · Distancia al stop (texto arriba + barra) · Apertura.
Anchos de referencia (contenido 1096 px): `120 80 44 76 76 96 156 68 minmax(190px,1fr) 84`, gap 10. Orden por defecto: distancia al stop ascendente (largas y cortas mezcladas).
- P&L: misma regla en largas y cortas: signo, flecha y color según si **gana o pierde**, no según hacia dónde fue el precio.
- Valor: siempre en absoluto. Cantidad: siempre positiva; el lado lo dice el badge.
- Recomendado: construir `PositionRow` como configuración de `DataTable` (columnas + renderers).

### `OrderCard` / `OrderRow` + `RejectReason`
- Rechazadas agrupadas arriba con encabezado "Rechazadas · N" (neg-soft); luego "Resto · N".
- `RejectReason`: si `reject_reason` → bloque neg-soft con motivo, detalle y "Broker: {raw}". Si `null` → bloque sunken con ícono punteado, "Motivo no disponible" + `HelpTip term="reject_reason_missing"`.
- En tabla, el motivo va en sub-fila debajo de la orden (no en otra pantalla).
- Vencida (`expired`): línea de explicación en la misma tarjeta o sub-fila, según `time_in_force`: `day` → "Orden del día: venció al cierre (16:00 ET) sin que el precio llegara al límite."; otros → "Venció el {expired_at} sin ejecutarse."
- Filtros: Todas · Rechazadas · Pendientes · Ejecutadas · Canceladas · Vencidas (chip con contador; deshabilitado si da 0).
- Hora: siempre con "ET".
- Recomendado: construir `OrderRow` como configuración de `DataTable` con sub-filas.

### `FunnelChart`
Barras horizontales escalonadas: largo = `reached / total`; dentro, 3 segmentos: pasaron (`--qa-chart-1`), están acá (`--qa-accent`), cayeron (`.qa-hatch`). Separador entre segmentos: `inset -2px 0 0 var(--qa-surface)`. Texto por fila: "26 llegaron · 1 acá · −11". Etapa con 0: pista vacía + "todavía ninguna".

### `DropReasons`
Por etapa: lista de motivos con conteo y barra (text-3). Teléfono: acordeón (`aria-expanded`), cerrado muestra el motivo principal. Escritorio: tarjetas abiertas en grilla de 3. Enlace "Ver las N descartadas" → filtra `HypothesisList`.

### `CycleSummaryCard`
Entraron / siguen / cayeron + paso por etapa + `headline`.

### `HypothesisCard`
Nombre, `Badge stage`, idea, métrica + fecha, chevrón. Descartada: nombre en text-2, `Badge discarded`, bloque sunken con motivo y detalle (valor vs umbral). Enlace a ficha.

### `HomeStatusBanner`
`level` ok → neutro "Todo en orden"; warning/critical → fondo soft + ícono + enlace.

### `FeedList` ("Desde tu última visita")
Ítems `FeedEvent`: glifo en cuadrado 28 px (color por severidad), título, subtítulo, chevrón; todo el ítem es enlace. Ver EVENTOS.md.

### `EventList` ("Próximos eventos")
Bloque de fecha (día de semana + número), título, `Badge` de tipo (Rebalanceo accent, Macro info, Feriado neutral), subtítulo, hora `ET · ART`.

---

## 4. Gráficos

Biblioteca sugerida para todo salvo velas: **Recharts** (simple, SVG, buen soporte de React). Alternativa si hace falta más control: visx. Velas: **lightweight-charts** con `candleTheme()` de `tokens.ts`.

| Componente | Notas |
|---|---|
| `EquityChart` | 3 series: cuenta (chart-1, continua, 2,25 px, área 7 %), SPY (chart-2, guiones `5 3`), sombra (chart-3, puntos `1.5 3`). Eje Y a la derecha, 3–4 marcas, `k`. Frase-resumen arriba ("En 3 meses la cuenta subió +4,7 %; SPY, +3,1 %"). Tooltip fijo al tocar en teléfono. |
| `CandleChart` | lightweight-charts. Alza hueca verde, baja llena roja, volumen al 35 %. MM20 en chart-4. Lineal/Log. Solo diario. |
| `CumulativeChart` | % desde el primer día; línea 0 % con `--qa-border-control`; etiqueta al final de cada serie con valor; trazos de `seriesDash`. |
| `HBar` / `DivergingBar` | Exposición: sub = chart-2 a la izquierda, sobre = chart-1 a la derecha, eje central border-strong, valor en pp. |
| `Heatmap` | `heatStep()` + `heatScale`; valor dentro de la celda; 4 columnas escritorio / 3 teléfono; ordenado de mayor a menor. |
| `Sparkline` | SVG propio (sin biblioteca). Trazo text-3/text-2, solo el punto final con color semántico. Sin ejes. |
| `RecessionBands` | `ReferenceArea` a todo el alto con `--qa-recession` + etiqueta arriba. |

Reglas comunes: grilla horizontal `--qa-grid`, sin grilla vertical (salvo velas), ejes en mono 11 px `--qa-axis`, unidades visibles, `role="img"` + `aria-label` con resumen textual, estados cargando (barras esqueleto) y sin datos (recuadro punteado + motivo + qué probar).

---

## 5. Accesibilidad (checklist por PR)

- [ ] Contraste: solo tokens; nada de opacidades sobre texto salvo deshabilitado.
- [ ] Toda variación tiene signo y flecha; todo estado tiene ícono + palabra.
- [ ] Objetivos táctiles ≥ 44 px (32 px en densidad compacta, solo escritorio).
- [ ] Foco visible (`:focus-visible`) en todo lo interactivo; orden de tabulación lógico.
- [ ] Controles reales (`button`, `a`, `input`), nunca `div` con `onClick`.
- [ ] Gráficos con `aria-label` de resumen.
- [ ] Zoom 200 % sin scroll horizontal; tamaños en rem.
- [ ] `prefers-reduced-motion` respetado.

---

## 6. Rendimiento (v0.3)

Referencia: canvas, página **Rendimiento** (tablero “Componentes nuevos”). Uso y datos: RENDIMIENTO.md.

### `SubjectHeader`
| Prop | Tipo | Nota |
|---|---|---|
| `subject` | `PerfSubject` | Define insignia, entorno e identificador |
| `period`, `asOf`, `rebalances` | de `PerformanceReport` | |
| `sentence` | `string` | Armada por el cliente con la plantilla de RENDIMIENTO.md §4.1 |
| `back?` | `{ label, href }` | Ausente en cartera (pestaña) |
| `range`, `availableRanges`, `onRangeChange` | | `SegmentedControl` YTD · 1A · 3A · 5A · Todo; opciones no disponibles deshabilitadas con motivo en `title` y `aria-describedby` |
| `compare`, `onCompareChange` | `{ type, id, name } \| 'spy' \| null` | Abre `CompareSelector` |
Teléfono: controles a todo el ancho, debajo de la frase. Escritorio: a la derecha.

### `CompareSelector`
Escritorio: menú; teléfono: hoja inferior. Opciones con `role="radio"`: SPY (por defecto), Nada, Otro backtest…, Otra estrategia en ejecución…, recientes. Al elegir “Otro…” abre la lista con búsqueda (`DataTable` con casillas, `maxSelected` = 1 en este contexto). Muestra siempre la regla “Se comparan como máximo dos”. En pantalla, los sujetos comparados se ven como fichas A/B con la muestra de trazo y botón “Quitar B” de 44 px.

### `ShortSampleBanner`
| Prop | Tipo |
|---|---|
| `sample` | `PerformanceReport['sample']` |
| `period` | para el texto (“3,6 meses y 15 rebalanceos”) |
| `collapsed?` | `boolean` (recordado por dispositivo con `localStorage`) |
Tono info (`--qa-info-soft`), ícono reloj de arena (pocos meses) o ciclo (pocos rebalanceos), `role="note"`. Plegado: botón de 44 px con `aria-expanded`.

### `MetricGroup` / `MetricRow`
- `MetricGroup`: `title`, `columns` (`['A']` o `['A','B']` con nombre y muestra de trazo), `rows: MetricRow[]`, `state: 'ready' | 'loading' | 'stale'`.
- `MetricRow`: `metricKey`, `label`, `help` (clave de help-texts), `a: MetricValue`, `b?: MetricValue`, `kind: 'variation' | 'level'`, `format: 'pct' | 'pp' | 'ratio' | 'count'`.
- `variation` → signo + flecha + color semántico. `level` → sin flecha ni color (Sharpe, volatilidad, beta…).
- `short_sample` → texto secundario + marca punteada “muestra corta”. `not_computable` → “—” + “no calculable” + `reason` en la línea secundaria.
- Grilla: `minmax(0,1fr) 92px 76px` (escritorio) / `minmax(0,1fr) 86px 62px` (teléfono); filas ≥ 44 px; columna B en texto secundario.

### `ChartCard`
Contenedor de todo gráfico de Rendimiento.
| Prop | Tipo |
|---|---|
| `title`, `help?` | |
| `summary` | `string` — siempre visible y usada como `aria-label` |
| `controls?` | `ReactNode` (segmentados propios) |
| `view`, `onViewChange` | `'chart' \| 'table'` (preferencia por dispositivo) |
| `table` | `{ columns, rows }` para `DataTable` |
| `state` | `'ready' \| 'loading' \| 'empty' \| 'error'` + `emptyMessage`, `onRetry` |
| `pinned?` | `{ label, values[] } \| null` — franja de tooltip fijo (teléfono) con botón “quitar” |

### `DrawdownChart`
Área bajo cero en `--qa-neg` al 14 % + línea `--qa-neg`; B en guiones `--qa-chart-2`. `highlight: number` (por defecto 3) peores caídas como bandas `--qa-neg` al 8 % numeradas en círculos (coinciden con la tabla). Eje Y de 0 hacia abajo con marcas cada 10 %. En backtest, franja de reserva fuera de muestra (`--qa-info` al 7 %).

### `CumulativeChart` (extensión)
Nueva prop `scale: 'linear' | 'log'`. En log, marcas en 0, +50, +100, +200, +300 % (factores 1; 1,5; 2; 3; 4). Línea 0 % con `--qa-border-control`.

### `MonthlyHeatmap`
| Prop | Tipo |
|---|---|
| `data` | `monthly_returns` |
| `annual` | `annual_returns` (columna Año y columna B) |
| `thresholds` | `[1, 3, 6]` (mensual) — `monthlyHeatStep()` |
| `layout` | `'grid' \| 'cards'` (automático por ancho) |
| `openYears?` | `3` en tarjetas |
- `grid` (≥ 1024 px útiles): fila por año, 12 celdas + Año + B; celdas 26 px de alto, número dentro.
- `cards` (< 1024 px): tarjeta por año con total y B; grilla 6 × 2 (4 × 3 por debajo de 360 px), celda ≥ 44 px con mes + valor; años viejos plegados (`aria-expanded`).
- Mes sin dato: celda punteada “—”. `role="table"`, cada celda con `aria-label` “mar 2020: −9,4 %”.

### `ReturnsBars`
`by: 'month' | 'year'`. Barras de A rellenas (`--qa-pos` / `--qa-neg`) con ▲/▼ y valor encima o debajo; B como contorno `--qa-text-3` al lado. Línea 0 con `--qa-border-control`. En teléfono, solo flecha (el valor exacto en “Tabla”); años abreviados (’15).

### `ReturnsHistogram`
`frequency: 'monthly' | 'daily'` con conmutador “Mensual / Diaria”; **mensual por defecto**. Diaria es vista secundaria, deshabilitada (con motivo vía `aria-describedby`) mientras `distributions.daily` sea `null` (menos de 63 días).
- Mensual: barras por intervalo de 1 %: negativas `--qa-neg`, positivas `--qa-pos`; B como escalón en guiones `--qa-chart-2`; línea vertical del promedio `--qa-text` con etiqueta “Promedio +1,0 %”. Ejes −10, −5, 0, +5, +10 %.
- Diaria: intervalos de 0,25 %; línea de VaR 95 % (guiones `--qa-text`) y CVaR (puntos), cola izquierda sombreada en `--qa-neg` al 7 %; frase con VaR y CVaR en palabras. Ejes −4 … +4 %.

### `ReturnsNotComputable` (v0.3.2)
Reemplaza métricas y gráficos cuando `status: 'returns_not_computable'`. Dos variantes según `not_computable_reason`:
- `unregistered_flows` (cartera, flag encendido): título “Retorno no calculable: hay N aporte(s) sin registrar”, qué movimiento (fecha y monto) y “cuando se registre, el informe se calcula solo”, más la tarjeta de valor en USD descrita abajo.
- `capital_not_isolated` (estrategia en ejecución): título “Rendimiento no aislable”, explicación (comparte posiciones y caja con la cuenta) y enlaces al backtest y a la cuenta. Sin tarjeta de valor ni números.

Variante `unregistered_flows`, detalle: Tarjeta punteada: ícono, “Retorno no calculable: hay aportes sin registrar”, explicación y enlace al manual. Debajo, `Card` “Valor de la cuenta” en USD marcado “incluye aportes”: línea neutra (`--qa-text-2`), sin %, sin colores de ganancia/pérdida, con los saltos de `flows.unreconciled` marcados (línea punteada `--qa-warn` + fecha y monto con “?”). Prop `unreconciled` cambia el texto a “Hay N movimientos sin registrar: …”.

### `RollingChart`
`metric: 'sharpe' | 'volatility'` (uno a la vez), `window: 126 | 252`. A continua, B en guiones; línea 0 marcada en Sharpe. Los primeros días sin ventana completa no se dibujan (sin interpolar).

### `CalcConventions`
Pie en `--qa-surface-sunken` con las convenciones del informe y enlace al manual (44 px en teléfono).
