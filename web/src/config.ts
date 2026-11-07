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
  stages: [
    { id: 'e0', code: 'E0', label: 'Admisión' },
    { id: 'e1', code: 'E1', label: 'Filtro' },
    { id: 'e2', code: 'E2', label: 'Validación en el motor' },
    { id: 'e3', code: 'E3', label: 'Holdout' },
  ],
  laterPhases: [
    { id: 'shadow', label: 'Sombra local' },
    { id: 'paper', label: 'Paper' },
    { id: 'candidate', label: 'Candidata' },
  ],
  defaultPeriodDays: 90,
  defaultCycle: 'all' as 'all' | 'latest',
  defaultPeriod: 'all' as '30d' | '90d' | 'all',
  minForChart: 2,
  cardsUpTo: 30,
  phoneInitialCards: 8,
  previousCyclesShown: 3,
} as const;

export const cycleConfig = {
  maxHypotheses: 5,
  milestones: { proposal: 1 },
  reviewAfterCycles: 6,
  reviewFeedEvent: true,
  cli: { open: 'quantagent registro ciclo abrir {ciclo} {hipótesis…}', close: 'quantagent registro ciclo cerrar {ciclo}', report: 'quantagent registro ciclo informe {ciclo}', streak: 'quantagent registro corte' },
  idPattern: /^\d{4}-(0[1-9]|1[0-2])$/,
} as const;

export const marketConfig = {
  staleAfterCloseMinutes: 180,
  regime: { maDays: 200, volDays: 20, volMedianDays: 252, stripMonths: 12 },
  enabledTop: 3,
  sectorSymbols: ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY'],
  m5Universe: ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'SPY', 'EEM', 'EWZ'],
  universe: { us_equity: ['SPY', 'QQQ', 'IWM'], sector: ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'], international: ['EFA', 'EEM', 'EWJ', 'EWZ', 'FXI'], bonds: ['SHY', 'IEF', 'TLT', 'TIP', 'AGG', 'LQD'], commodities: ['GLD', 'SLV', 'DBC', 'USO'], currency: ['FXE'], real_estate: ['VNQ'] },
  macroStale: { graceBusinessDays: 2, fallbackFrequencyMultiplier: 1.5 },
  referenceSymbols: ['SPY', 'EEM', 'EWZ'],
  heatThresholds: { d1: [0.25, 0.75, 1.5], w1: [0.5, 1.5, 3], m1: [1, 3, 6], m3: [2, 5, 10], ytd: [3, 8, 15] },
  minSessions: { d1: 2, w1: 6, m1: 21, m3: 63, ytd: 0 },
  macroRanges: ['1y', '5y', 'all'] as const,
  defaultMacroRange: { phone: '5y', desktop: 'all' } as const,
  macroSeries: [{ code: 'T10Y2Y', active: true }, ...['CPIAUCSL', 'UNRATE', 'PAYEMS', 'INDPRO', 'BAMLH0A0HYM2', 'VIXCLS', 'T5YIE', 'DFII10', 'DTWEXBGS'].map(code => ({ code, active: false }))],
  maxCompare: 5, defaultCompare: ['SPY'], defaultCompareRange: '1y' as '1m' | '3m' | '1y',
  candleRanges: ['3m', '6m', '1y'] as const, eventsDays: 14, eventKinds: ['macro', 'fomc', 'holiday', 'early_close'] as const,
} as const;

export function sectorHeatStep(pct: number, key: keyof typeof marketConfig.heatThresholds): number {
  const [a, b, c] = marketConfig.heatThresholds[key];
  if (pct <= -c) return 0;
  if (pct <= -b) return 1;
  if (pct < -a) return 2;
  if (pct <= a) return 3;
  if (pct < b) return 4;
  if (pct < c) return 5;
  return 6;
}

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

export const performanceConfig = {
  minDaysForReport: 21, dailyDistributionMinDays: 63,
  defaultDistribution: 'monthly' as 'monthly' | 'daily',
  minDays: { cagr: 252, calmar: 252, alpha: 252, volatility: 63, sharpe: 63, sortino: 63, beta: 63 },
  shortSampleUntilDays: 252, minRebalances: 12, maxSubjects: 2,
  defaultRange: 'all' as 'ytd' | '1y' | '3y' | '5y' | 'all', defaultCompare: 'spy',
  rollingWindows: [126, 252] as const, monthlyHeatThresholds: [1, 3, 6] as const,
  heatmapOpenYearsPhone: 3, retrySeconds: 30,
} as const;

export function monthlyHeatStep(pct: number): number {
  const [a, b, c] = performanceConfig.monthlyHeatThresholds;
  if (pct <= -c) return 0;
  if (pct <= -b) return 1;
  if (pct < -a) return 2;
  if (pct <= a) return 3;
  if (pct < b) return 4;
  if (pct < c) return 5;
  return 6;
}

export const featureFlags = { accountPerformanceTab: true, runPerformance: true } as const
