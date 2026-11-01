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

/**
 * Una cuenta puede existir en la configuración sin datos todavía
 * (por ejemplo, Live antes de abrirse). En ese caso NO se mandan ceros:
 * se manda `status: 'no_data'` y el cliente muestra el estado "Sin datos todavía".
 */
export type AccountSummary = AccountSummaryActive | AccountSummaryNoData;

export interface AccountSummaryNoData extends AccountRef {
  status: 'no_data';
  /** Por qué no hay datos. 'unknown' si el backend no lo puede distinguir. */
  no_data_reason: 'not_opened' | 'not_funded' | 'unknown';
  /** Última vez que el backend consultó al broker por esta cuenta. */
  checked_at: string;
  /** Próxima consulta programada (opcional). */
  next_check_at: string | null;
}

export interface AccountSummaryActive extends AccountRef, Freshness {
  status: 'active';
  equity: number;
  /** Valor de mercado de las posiciones largas (positivo). */
  long_value: number;
  /** Valor de mercado de las posiciones cortas, en absoluto (positivo). */
  short_value: number;
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

export type PositionSide = 'long' | 'short';

export interface Position {
  symbol: string;
  description: string | null;         // 'ETF S&P 500'
  side: PositionSide;
  /** Siempre positiva. El broker informa negativa en cortas: el backend la normaliza. */
  qty: number;
  avg_entry_price: number;
  current_price: number;
  /** En absoluto (positivo) también en cortas. */
  market_value: number;
  /** Ya con el signo correcto según el lado: en una corta, gana si el precio baja. */
  unrealized_pl: number;
  unrealized_pl_pct: number;
  /** Larga: por debajo del precio. Corta: por encima. null → "Sin stop vigente". */
  stop_price: number | null;
  opened_at: string;
}

export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit';
/** Ortografía del backend: 'cancelled' con dos L. */
export type OrderStatus = 'filled' | 'partially_filled' | 'pending' | 'rejected' | 'cancelled' | 'expired';
export type TimeInForce = 'day' | 'gtc' | 'ioc' | 'fok' | 'opg' | 'cls';

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
  /** Para explicar una vencida ("orden del día: venció al cierre"). */
  time_in_force: TimeInForce | null;
  submitted_at: string;
  cancelled_at: string | null;
  expired_at: string | null;
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
export interface SectorExposure { sector: string; long: number; short: number; pct: number; }
export interface Dividend { date: string; symbols: string[]; net: number; withheld: number; }

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
  level: 'ok' | Severity | 'neutral';
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
  source: 'alpaca' | 'strategy' | 'fred' | 'manual';
  account_id: string | null;
}

export interface MarketClock {
  is_open: boolean;
  next_open: string;
  next_close: string;
}

export type PerfRange = 'ytd' | '1y' | '3y' | '5y' | 'all'
export type MetricKey = 'total_return' | 'cagr' | 'positive_months' | 'best_month' | 'worst_month' | 'volatility' | 'sharpe' | 'sortino' | 'max_drawdown' | 'calmar' | 'alpha' | 'beta' | 'relative_return'
export interface MetricValue { value: number | null; status: 'ok' | 'short_sample' | 'not_computable'; reason: string | null; detail?: { month?: string; positive?: number; total?: number; peak_date?: string; trough_date?: string; recovery_date?: string | null; days?: number } }
export interface BacktestListItem { id: string; label: string; hypothesis_name: string | null; start: string; end: string; total_return: number }
export interface PerformanceReport {
 subject: { type: 'backtest'|'run'|'portfolio'; id: string; name: string; env: Env|null; run_label: string|null; account_id: string|null; hypothesis_id: string|null; backtest_id: string|null; out_of_sample_start: string|null; costs_bps: number|null; strategies: {run_id:string;name:string;total_return:number}[]|null }
 benchmark: {symbol:'SPY'}|null; status:'ready'|'insufficient_data'|'no_runs'|'returns_not_computable'; not_computable_reason:'unregistered_flows'|'capital_not_isolated'|null; range:PerfRange; available_ranges:{range:PerfRange;available:boolean;reason:string|null}[]; period:{start:string;end:string;trading_days:number;complete_months:number;rebalances:number|null}; as_of:string; stale:boolean; aligned_with:string|null
 sample:{short:boolean;reasons:('few_days'|'few_months'|'few_rebalances')[];not_computable:MetricKey[];unreliable:MetricKey[];months_until_full:number|null;thresholds:{descriptive_days:number;risk_days:number;annualized_days:number;min_rebalances:number}}
 flows:{registered:boolean;unreconciled:{date:string;amount:number}[]}|null; insufficient:{days_have:number;days_needed:number;first_report_estimate:string|null}|null; no_runs:{approved_at:string;account_id:string;next_rebalance_at:string|null}|null
 headline:{total_return:number;benchmark_total_return:number|null;max_drawdown:number;max_dd_trough:string;max_dd_recovery_days:number|null;shadow_gap_pp:number|null}|null
 metrics:Record<MetricKey,MetricValue>; benchmark_metrics:Record<MetricKey,MetricValue>
 series:{equity:{date:string;value:number;benchmark:number|null;shadow:number|null}[];drawdown:{date:string;dd:number;benchmark_dd:number|null}[]}
 rolling:{window_days:126|252;points:{date:string;sharpe:number|null;volatility:number|null;benchmark_sharpe:number|null;benchmark_volatility:number|null}[]}[]
 monthly_returns:{year:number;month:number;ret:number;benchmark_ret:number|null;complete:boolean}[]; annual_returns:{year:number;ret:number;benchmark_ret:number|null;partial:boolean}[]
 distributions:{monthly:{bin_width_pct:number;bins:{from:number;to:number;count:number;benchmark_count:number|null}[];mean:number;benchmark_mean:number|null;n:number};daily:({bin_width_pct:number;bins:{from:number;to:number;count:number;benchmark_count:number|null}[];mean:number;benchmark_mean:number|null;n:number;var_95:number;cvar_95:number})|null;daily_reason:string|null}
 worst_drawdowns:{rank:number;peak_date:string;trough_date:string;recovery_date:string|null;depth:number;days_to_trough:number;days_to_recovery:number|null;total_days:number}[]; cash_flows:{date:string;amount:number}[]|null
 conventions:{frequency:'daily';method:'time_weighted';annualization_days:number;risk_free_source:string;risk_free_avg:number;benchmark:string;data_source:string;costs_bps:number|null;manual_url:string}
}
