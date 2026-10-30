# Inventario de componentes

Referencia visual: canvas, página **Sistema de diseño**, hojas 03 (Componentes A), 04 (Componentes B) y 05 (Gráficos).
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
| `canceled` | transparente + borde control / text-2 | círculo tachado |
| `paper` | borde punteado control / text-2 | — |
| `live` | text / surface (invertido) | punto, texto "Live · dinero real" |
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

### `DateStepper`
`‹ Hoy · jue 29/10 ›`. Flechas de 44 px; "siguiente" deshabilitada en hoy. Tocar la fecha abre un calendario.

### `BottomNav` (< 1024 px) / `SideNav` (≥ 1024 px)
4 destinos: Inicio, Cuenta, Embudo, Mercado. Ícono + texto siempre. Activo: `aria-current="page"`, accent-soft. Destinos aún no construidos: texto apagado + "pronto". SideNav: ícono de advertencia junto a "Cuenta" si hay alarmas. Densidad y tema al pie (escritorio) o en el ícono de ajustes de Inicio (teléfono).

---

## 3. Dominio

### `MetricCard`
Etiqueta + `Num` + `Delta` + `Sparkline?` + pie (hora). Estados de la hoja 03 (incluye "próximamente" y "desactualizado").

### `AccountCard` (Inicio)
`AccountSummary` → nombre, badge entorno, valor, hoy, semana, sparkline 30 días, alarma (`Notice` compacto) o "Sin alarmas · {meta}". Toda la tarjeta es enlace a Cuenta.

### `AccountSummaryCard` (Cuenta)
Valor (`num-xl`), hoy y semana (`Delta stacked`), pie: invertido / efectivo / órdenes de hoy.

### `StopDistanceBar`
| Prop | Tipo |
|---|---|
| `price`, `stop`, `qty` | `number`, `number \| null`, `number` |
| `variant?` | `'compact' \| 'anatomy'` |
- Cálculo en `config.ts → stopDistance()`. Escala fija 0–`scalePct`.
- Zona `ok`: relleno text-2. `attention`: warn + triángulo + "Atención". `near`: neg + octógono + "Cerca". `none`: pista punteada + "Sin stop vigente".
- Texto siempre presente: "Cerca · 1,5 % · USD 125". `role="meter"` con `aria-valuenow`, `aria-valuetext`.
- Supuesto: solo posiciones largas (ver DECISIONES.md).

### `PositionCard` (teléfono) / `PositionRow` (≥ 1024 px)
Card: símbolo + descripción · valor / `qty × precio · entr.` · P&L (`Delta`) / `StopDistanceBar compact`.
Row: columnas Símbolo · Cant. · Entrada · Actual · Valor · P&L no realizado · Stop · Distancia al stop (texto arriba + barra) · Apertura.
Anchos de referencia (contenido 1096 px): `150 48 80 80 100 168 72 minmax(200px,1fr) 84`, gap 12. Orden por defecto: distancia al stop ascendente.

### `OrderCard` / `OrderRow` + `RejectReason`
- Rechazadas agrupadas arriba con encabezado "Rechazadas · N" (neg-soft); luego "Resto · N".
- `RejectReason`: si `reject_reason` → bloque neg-soft con motivo, detalle y "Broker: {raw}". Si `null` → bloque sunken con ícono punteado, "Motivo no disponible" + `HelpTip term="reject_reason_missing"`.
- En tabla, el motivo va en sub-fila debajo de la orden (no en otra pantalla).
- Hora: siempre con "ET".

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
