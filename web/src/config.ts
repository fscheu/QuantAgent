/**
 * Parámetros de producto que el diseño usa pero que todavía no están decididos.
 * Cambiar acá, nunca dentro de un componente. Ver DECISIONES.md.
 */

export const stopConfig = {
  /** Escala visual de la barra: 0 % (en el stop) a este % (barra llena). */
  scalePct: 15,
  /** Menos que esto = "Cerca del stop" (negativo + octógono). PROVISORIO */
  nearPct: 2,
  /** Menos que esto = "Atención" (advertencia + triángulo). PROVISORIO */
  attentionPct: 5,
} as const;

export type PositionSide = 'long' | 'short';
export type StopZone = 'breached' | 'near' | 'attention' | 'ok' | 'none';

export interface StopDistance {
  zone: StopZone;
  /** % de distancia al stop. Negativo solo si el precio ya pasó el stop ('breached'). */
  pct: number | null;
  /** Margen en USD (|precio − stop| × cantidad). En 'breached' es cuánto lo pasó. */
  amount: number | null;
  /** 0–1: largo del relleno de la barra (escala fija 0–scalePct). */
  fill: number;
  /** La barra crece desde la izquierda (larga: stop abajo) o desde la derecha (corta: stop arriba). */
  anchor: 'left' | 'right';
}

/**
 * Larga: stop debajo del precio → distancia = (precio − stop) / precio.
 * Corta: stop encima del precio → distancia = (stop − precio) / precio.
 * Las zonas se leen igual en los dos lados. `qty` siempre positiva.
 */
export function stopDistance(side: PositionSide, price: number, stop: number | null, qty: number): StopDistance {
  const anchor = side === 'short' ? 'right' : 'left';
  if (stop == null) return { zone: 'none', pct: null, amount: null, fill: 0, anchor };
  const pct = ((side === 'short' ? stop - price : price - stop) / price) * 100;
  const amount = Math.abs(price - stop) * Math.abs(qty);
  if (pct <= 0) return { zone: 'breached', pct, amount, fill: 0, anchor };
  const zone: StopZone = pct < stopConfig.nearPct ? 'near' : pct < stopConfig.attentionPct ? 'attention' : 'ok';
  const fill = Math.min(1, pct / stopConfig.scalePct);
  return { zone, pct, amount, fill, anchor };
}

/** P&L no realizado según el lado (por si el backend no lo manda calculado). */
export function unrealizedPl(side: PositionSide, entry: number, price: number, qty: number) {
  const perShare = side === 'short' ? entry - price : price - entry;
  return { amount: perShare * Math.abs(qty), pct: (perShare / entry) * 100 };
}

export const accountConfig = {
  /** Cada cuánto el backend vuelve a consultar una cuenta sin datos (minutos). PROVISORIO */
  noDataRecheckMinutes: 15,
} as const;

export const dataConfig = {
  /** Un dato se marca "desactualizado" si supera esto (minutos), en horario de mercado. PROVISORIO */
  staleMinutesMarketOpen: 30,
  /** Fuera de horario de mercado. PROVISORIO */
  staleMinutesMarketClosed: 24 * 60,
  /** Frecuencia de refresco en el cliente (segundos). PROVISORIO */
  pollSecondsMarketOpen: 60,
  pollSecondsMarketClosed: 900,
} as const;

export const screenConfig = { showComingSoon: true } as const;

export const funnelConfig = {
  /** Orden canónico de etapas. Nombres de 0–3 PROVISORIOS. */
  stages: [
    { id: 'e0', label: 'Etapa 0' },
    { id: 'e1', label: 'Etapa 1' },
    { id: 'e2', label: 'Etapa 2' },
    { id: 'e3', label: 'Etapa 3' },
    { id: 'reserve', label: 'Reserva' },
    { id: 'shadow', label: 'Sombra' },
    { id: 'paper', label: 'Paper' },
    { id: 'candidate', label: 'Candidata' },
    { id: 'production', label: 'Producción' },
  ],
  /** Ciclo por defecto al abrir Embudo: 'all' | 'latest'. PROVISORIO */
  defaultCycle: 'all' as 'all' | 'latest',
  /** Período por defecto (días). */
  defaultPeriodDays: 90,
} as const;

export const tableConfig = {
  /** Debajo de este ancho, DataTable se muestra como tarjetas. */
  cardBreakpoint: 1024,
  /** Filas esqueleto mientras carga. */
  skeletonRows: 5,
  /** Máximo de filas tildadas cuando la selección alimenta un gráfico (series simultáneas). */
  maxSelectedForChart: 5,
} as const;

export const homeConfig = {
  /** Máximo de ítems en "Desde tu última visita" antes de "Ver todo". */
  feedMaxItems: 6,
  /** Ventana si no hay última visita registrada (horas). */
  feedFallbackHours: 24,
  /** Horizonte de "Próximos eventos" (días). */
  calendarDays: 14,
  /** Severidades del feed, de mayor a menor prioridad. */
  feedSeverityOrder: ['critical', 'warning', 'info', 'neutral'] as const,
} as const;

export const featureFlags = { accountPerformanceTab: true, runPerformance: true } as const
