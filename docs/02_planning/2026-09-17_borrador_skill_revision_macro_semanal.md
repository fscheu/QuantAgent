# Borrador: skill Hermes "revisión macro semanal" (capa L1)

Fecha: 2026-09-17 · Estado: **borrador, no instalado** (instalar en `~/.hermes/skills/` requiere confirmación de Fede)
Contexto: `2026-09-16_retrospectiva_autodev_y_plan_oficina_inversion.md` §3, capa L1

---

## 1. Alcance

**Objetivo:** que Fede lea cada lunes una nota de una página con el estado del régimen macro global y una
postura tentativa por región y sector, con fuentes citadas y preguntas abiertas. Es **soporte de decisión y
aprendizaje**, no una recomendación operativa. No mueve dinero, no genera órdenes, no toca QuantAgent.

**Produce:** una nota en el vault + un resumen de 8 líneas por Telegram.

**Primera versión (4 semanas):** solo la nota. Sin scoring numérico, sin histórico automático, sin señales
alternativas (L6). Cuando haya 4 notas, se revisa qué preguntas se repiten y ahí se decide si algo se
convierte en software.

**Prerrequisito verificado 2026-09-16:** `hermes doctor` muestra `web` ✓ (búsqueda/extracción disponible).
`x_search` falta (`XAI_API_KEY`), no se necesita para V1.

## 2. Decisiones que Fede tiene que confirmar

1. Carpeta destino. Propuesta: `02_Areas/Inversiones/Macro Semanal/` (Area nueva; hoy no existe).
   Alternativa: `03_Resources/Weekly/`.
2. Horario: domingos 18:00 ART (nota lista para el lunes).
3. Fuentes iniciales (§5). Agregar o quitar.
4. Regiones y sectores a cubrir en V1 (§4, plantilla).

## 3. SKILL.md propuesto

```markdown
---
name: revision-macro-semanal
description: "Weekly macro regime review for Fede: rates, FX, equity indices, commodities, geopolitics → regional/sector stance note in the Obsidian vault + Telegram summary. Decision support, not trade signals."
version: 0.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [fede, inversiones, macro, weekly, obsidian, decision-support]
    related_skills: [obsidian-skill-fede, fede-ops, pm-agent]
---

# Revisión macro semanal

## Propósito
Producir cada semana UNA nota en el vault con el estado del régimen macro global y una postura tentativa
por región y sector, para que Fede aprenda a leer el contexto antes de decidir asignación de cartera.
Es soporte de decisión. No es asesoramiento financiero ni una orden. Nunca escribe en QuantAgent.

## Cuándo usar
- Cron dominical (job "Revisión Macro Semanal").
- Fede pide "revisión macro", "cómo viene el mercado esta semana", "actualizá la postura".

## Flujo
1. Fecha de la nota: el lunes siguiente (YYYY-MM-DD). Si ya existe la nota de esa semana, actualizarla, no duplicar.
2. Leer la nota de la semana anterior (si existe) para arrastrar la postura y las preguntas abiertas.
3. Recolectar, con la tool web, y SIEMPRE con URL y fecha por dato:
   - Tasas: Fed funds y expectativas (FedWatch), ECB, BCRA (tasa de política), 10Y US, 10Y DE.
   - Monedas: DXY, EUR/USD, USD/BRL, USD/ARS oficial y CCL, brecha.
   - Índices: S&P 500, Nasdaq, Stoxx 600, Nikkei, MSCI EM, Merval (variación semanal y YTD).
   - Commodities: Brent, oro, cobre, soja.
   - Volatilidad y crédito: VIX, spreads HY.
   - 3 a 5 noticias políticas/geopolíticas de la semana con impacto de mercado plausible.
   - Calendario de la semana que empieza: datos macro y reuniones de bancos centrales.
4. Escribir la nota con la plantilla de abajo, usando obsidian-skill-fede (nunca escritura ad hoc).
5. Enviar a Telegram un resumen de 8 líneas: régimen, 3 movimientos clave, postura, 1 pregunta abierta, link a la nota.

## Reglas
- Cada número lleva fuente y fecha. Sin fuente, no se escribe el número.
- Separar explícitamente HECHOS (con fuente) de INTERPRETACIÓN (propia) y de POSTURA (tentativa).
- Postura por región/sector en 3 niveles: sobre-ponderar / neutral / sub-ponderar, con una frase de motivo.
- Marcar incertidumbre: si dos fuentes discrepan, decirlo.
- Prohibido: recomendar instrumentos concretos, sizing, o "comprar/vender ahora".
- Idioma: español rioplatense. Una página. Sin relleno.

## Plantilla de nota
---
type: macro-weekly
date: {YYYY-MM-DD}
status: draft
tags: [inversiones, macro, weekly]
---
# Revisión macro semanal — {YYYY-MM-DD}

## 1. Régimen en una frase
## 2. Tablero (variación semanal, fuente por fila)
| Variable | Valor | Δ semana | Fuente |
## 3. Qué pasó (hechos, 5 bullets con fuente)
## 4. Cómo lo leo (interpretación, 5 bullets)
## 5. Postura tentativa
| Región / sector | Postura | Motivo en una frase |
## 6. Cambios respecto de la semana pasada
## 7. Semana que viene (calendario)
## 8. Tres preguntas abiertas para Fede
## Fuentes
```

## 4. Regiones y sectores V1 (a confirmar)

Regiones: EE.UU., Europa, Japón, Emergentes, Argentina.
Sectores (EE.UU.): tecnología, financieras, energía, salud, consumo, industriales.

## 5. Fuentes iniciales (a confirmar)

- Datos: FRED, CME FedWatch, investing.com / tradingeconomics (tablero), BCRA (tasa, reservas), Ámbito/Rava (CCL, brecha).
- Noticias: Reuters, FT, Bloomberg (titulares), Ámbito, La Nación (Argentina).
- Calendario: tradingeconomics calendar, Fed/ECB calendars.

## 6. Cron (después de instalar el skill)

```bash
hermes cron create "0 18 * * 0" \
  "Ejecutá la revisión macro semanal completa siguiendo el skill revision-macro-semanal: recolectar datos con fuentes, escribir la nota de la semana en el vault y enviar el resumen." \
  --name "Revisión Macro Semanal" \
  --skill revision-macro-semanal --skill obsidian-skill-fede \
  --workdir /home/azureuser/repos/projects/claude-second-brain \
  --deliver telegram:-1003401012237:43
```

Verificar la zona horaria del schedule con `hermes cron list` tras crearlo (los jobs existentes muestran
"Next run" en -03:00; si el schedule se interpreta en UTC, usar `0 21 * * 0`).
El `chat_id`/topic de Telegram es el mismo que usa "Ideas de Proyectos"; confirmar si Fede quiere otro topic.

## 7. Verificación de Fede (10 min los lunes)

Leer la nota. Anotar una pregunta que le dejó (sección 8) respondiendo el Telegram. Al cabo de 4 notas:
¿qué preguntas se repiten? Eso define L2 (asignación).
