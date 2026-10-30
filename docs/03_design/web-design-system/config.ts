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

export type StopZone = 'near' | 'attention' | 'ok' | 'none';

/** Posiciones largas: distancia = (precio − stop) / precio. */
export function stopDistance(price: number, stop: number | null, qty: number) {
  if (stop == null) return { zone: 'none' as StopZone, pct: null, amount: null, fill: 0 };
  const pct = ((price - stop) / price) * 100;
  const amount = (price - stop) * qty;
  const zone: StopZone = pct < stopConfig.nearPct ? 'near' : pct < stopConfig.attentionPct ? 'attention' : 'ok';
  const fill = Math.max(0, Math.min(1, pct / stopConfig.scalePct));
  return { zone, pct, amount, fill };
}

export const dataConfig = {
  /** Un dato se marca "desactualizado" si supera esto (minutos), en horario de mercado. PROVISORIO */
  staleMinutesMarketOpen: 30,
  /** Fuera de horario de mercado. PROVISORIO */
  staleMinutesMarketClosed: 24 * 60,
  /** Frecuencia de refresco en el cliente (segundos). PROVISORIO */
  pollSecondsMarketOpen: 60,
  pollSecondsMarketClosed: 900,
} as const;

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

export const homeConfig = {
  /** Máximo de ítems en "Desde tu última visita" antes de "Ver todo". */
  feedMaxItems: 6,
  /** Ventana si no hay última visita registrada (horas). */
  feedFallbackHours: 24,
  /** Horizonte de "Próximos eventos" (días). */
  calendarDays: 14,
} as const;
