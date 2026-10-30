/**
 * Contrato de datos que necesitan las pantallas (propuesta).
 * snake_case para coincidir con Pydantic/FastAPI. Montos en USD como number.
 * Fechas en ISO 8601 UTC; el cliente las muestra en ET o ART (format.ts).
 *
 * Los campos `| null` son casos que el diseño YA contempla con un estado propio:
 * no los conviertan en 0 ni en string vacío.
 */

/* ---------- Comunes ---------- */
export type Env = 'paper' | 'live';
export type Severity = 'critical' | 'warning' | 'info';

export interface Freshness {
  /** Última vez que el backend leyó este dato del broker o del sistema. */
  as_of: string;
  /** El backend lo marca si supera el umbral; el cliente igual puede recalcular. */
  stale: boolean;
}

/* ---------- Cuentas (Cuenta, Inicio) ---------- */
export interface AccountRef {
  id: string;            // 'paper-m4'
  name: string;          // 'Paper M4'
  env: Env;
}

export interface AccountSummary extends AccountRef, Freshness {
  equity: number;
  invested: number;
  cash: number;
  day_change: number;
  day_change_pct: number;
  week_change: number;
  week_change_pct: number;
  positions_count: number;
  orders_today: number;
  orders_rejected_today: number;
  alarms: Alarm[];                    // vacío = sin alarmas
  spark_30d: number[] | null;         // mini gráfico de Inicio
}

export interface Position {
  symbol: string;
  description: string | null;         // 'ETF S&P 500'
  qty: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_pl_pct: number;
  stop_price: number | null;          // null → "Sin stop vigente"
  opened_at: string;
}

export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit';
export type OrderStatus = 'filled' | 'partially_filled' | 'pending' | 'rejected' | 'canceled';

export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  qty: number;
  limit_price: number | null;
  stop_price: number | null;
  status: OrderStatus;
  filled_qty: number;
  filled_avg_price: number | null;
  submitted_at: string;
  /** Motivo en lenguaje claro (traducido por el backend). null → "Motivo no disponible". */
  reject_reason: string | null;
  /** Explicación con datos ("necesitaba USD 2.742,00; había USD 1.910,43"). */
  reject_detail: string | null;
  /** Texto original del broker, se muestra chico debajo. */
  reject_reason_raw: string | null;
}

export interface Alarm {
  id: string;
  account_id: string | null;
  severity: Severity;
  kind: 'drawdown_limit' | 'shadow_divergence' | 'data_stale' | string;
  title: string;                       // 'Sombra desalineada'
  message: string;                     // 'El broker y la simulación local difieren 0,8 %.'
  since: string;
}

/* Planeado (marcado "Próximamente" en el diseño) */
export interface EquityPoint { date: string; account: number; benchmark: number | null; shadow: number | null; }
export interface SectorExposure { sector: string; account_weight: number; benchmark_weight: number; }
export interface Dividend { symbol: string; paid_at: string; gross: number; withheld: number; net: number; }

/* ---------- Embudo ---------- */
export type StageId = 'e0' | 'e1' | 'e2' | 'e3' | 'reserve' | 'shadow' | 'paper' | 'candidate' | 'production';

export interface FunnelStage {
  stage: StageId;
  reached: number;       // llegaron a la etapa
  here: number;          // están ahora en la etapa
  dropped: number;       // cayeron en la etapa
  // passed = reached − here − dropped (lo calcula el cliente)
}

export interface DropReason { stage: StageId; reason: string; count: number; }

export interface Funnel {
  cycle: number | 'all';
  period_days: number | null;
  total: number;
  stages: FunnelStage[];
  drop_reasons: DropReason[];
}

export interface CycleSummary {
  cycle: number;
  started_at: string;
  ended_at: string | null;
  entered: number;
  active: number;
  dropped: number;
  reached_by_stage: Partial<Record<StageId, number>>;
  headline: string | null;  // 'La mayoría cayó en etapa 1…' (opcional, generado)
}

export interface Hypothesis {
  id: string;
  name: string;
  idea: string;                         // una línea
  cycle: number;
  stage: StageId;                       // etapa actual o donde cayó
  status: 'active' | 'discarded';
  last_metric: { name: string; value: number; threshold: number | null } | null;
  discard_reason: string | null;
  discard_detail: string | null;        // '0,31 · umbral 0,50'
  updated_at: string;
}

/* ---------- Inicio ---------- */
export interface HomeStatus {
  level: 'ok' | Severity;
  headline: string;                     // 'Casi todo bien: 1 advertencia'
  message: string;
  link: string | null;
}

/** Ver EVENTOS.md para el catálogo de tipos. */
export interface FeedEvent {
  id: string;
  type: string;
  severity: Severity | 'neutral';
  occurred_at: string;
  account_id: string | null;
  title: string;
  subtitle: string | null;
  link: string;                         // ruta interna: '/cuenta/paper-m4?tab=ordenes'
  group_count: number;                  // >1 si agrupa varios hechos
}

export type CalendarKind = 'rebalance' | 'macro' | 'fomc' | 'holiday' | 'early_close' | 'own';

export interface CalendarEvent {
  id: string;
  kind: CalendarKind;
  starts_at: string | null;             // null = todo el día
  date: string;                         // 'YYYY-MM-DD' (fecha de NY)
  title: string;
  subtitle: string | null;
  source: 'alpaca' | 'fred' | 'manual' | 'system';
  account_id: string | null;
}

export interface MarketClock {
  is_open: boolean;
  next_open: string;
  next_close: string;
}
