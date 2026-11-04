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
  cash?: number | null;
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
  day_change: number | null;
  day_change_pct: number | null;
  week_change: number | null;
  week_change_pct: number | null;
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
  status: 'executed' | 'pending';
  /** Siempre positiva. El broker informa negativa en cortas: el backend la normaliza. */
  qty: number;
  avg_entry_price: number | null;
  current_price: number | null;
  /** En absoluto (positivo) también en cortas. */
  market_value: number | null;
  /** Ya con el signo correcto según el lado: en una corta, gana si el precio baja. */
  unrealized_pl: number | null;
  unrealized_pl_pct: number | null;
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

/* ---------- Embudo (v0.4: registro real de hipótesis) ---------- */
/** Id de ciclo mensual 'YYYY-MM'. `null` = historial importado sin ciclo. */
export type CycleId = string;
/** Etapas del ciclo. La reserva 2023+ NO es etapa (ver Hypothesis.used_reserve). */
export type CycleStageId = 'e0' | 'e1' | 'e2' | 'e3';
/** Fase posterior: fuera del ciclo, sin métricas de etapa. */
export type LaterPhaseId = 'shadow' | 'paper' | 'candidate';
export type StageId = CycleStageId | LaterPhaseId;
/** Veredictos reales del motor. ERROR y ALARMA no son caídas por mérito ("detenidas"). */
export type Verdict = 'PASA' | 'CORTA' | 'ERROR' | 'ALARMA';
/**
 * Estado derivado de la hipótesis (lo calcula el backend):
 * active     = pasó la última etapa evaluada y espera la siguiente
 * promoted   = pasó E3 con PASA (único logro que cuenta para la regla de revisión); puede seguir en fase posterior
 * discarded  = CORTA en alguna etapa
 * held       = ERROR o ALARMA (detenida; se revisa aparte, no suma a "cayeron")
 */
export type HypothesisStatus = 'active' | 'promoted' | 'discarded' | 'held';

export interface StageCounts {
  stage: CycleStageId;
  reached: number;          // llegaron
  here: number;             // están ahora (PASA de la etapa anterior, esperando o en curso)
  passed: number;           // PASA en esta etapa
  cut: number;              // CORTA
  error: number;            // ERROR (aparte)
  alarm: number;            // ALARMA (aparte)
}
export interface LaterPhaseCounts { phase: LaterPhaseId; reached: number; here: number; }

/** Valor contra umbral: '0,31 · umbral 0,50'. `value`/`threshold` null si el motor no lo calculó. */
export interface Measure { metric: string; value: number | null; threshold: number | null; unit: 'ratio' | 'pct' | 'count' | 'years' | null; comparator: '>=' | '<=' | null; }

export interface DropReason {
  stage: CycleStageId;
  verdict: Extract<Verdict, 'CORTA' | 'ERROR' | 'ALARMA'>;
  cause_code: string;       // agrupa: 'sharpe_below', 'few_trades', 'max_dd', 'history_short', 'control_mismatch', 'data_load', ...
  cause_label: string;      // 'Sharpe bajo el umbral'
  count: number;
  example: Measure | null;  // un caso representativo
}

export interface Funnel {
  /** 'all' = todos los ciclos mensuales (no incluye historial); null = "Sin ciclo" (el backend devuelve HistorySummary en su lugar). */
  cycle: CycleId | 'all';
  /** 30d/90d: hipótesis con algún evento de etapa en los últimos N días; su estado es el del último evento dentro de la ventana. */
  period: '30d' | '90d' | 'all';
  total: number;
  stages: StageCounts[];            // siempre e0..e3, aunque estén en cero
  later: LaterPhaseCounts[];        // shadow, paper, candidate
  promoted: number;
  drop_reasons: DropReason[];
  /** true si total < funnelConfig.minForChart: el cliente muestra el recorrido, no el embudo. */
  sample_too_small: boolean;
}

export interface CycleMilestone { key: 'proposal' | 'funnel_start' | 'paper_review' | 'close'; day: number | null; date: string; done: boolean; }

export interface CycleSummary {
  cycle: CycleId;                    // '2026-11'
  status: 'open' | 'closed';
  started_at: string;
  closed_at: string | null;          // null si abierto
  entered: number;                   // ≤ cycleConfig.maxHypotheses
  active: number;
  cut: number;
  held: number;
  promoted: number;
  reached_by_stage: Record<CycleStageId, number | null>;  // null = etapa no empezada (ciclo abierto) → "pendiente"
  milestones: CycleMilestone[];
  headline: string | null;           // frase generada; si null el cliente arma una con los conteos
  receipt_id: string | null;         // null si abierto
}

export interface ReviewRule {
  limit: number;                     // 6
  streak: number;                    // ciclos CERRADOS seguidos sin promovida
  reached: boolean;
  last_cycles: { cycle: CycleId; promoted: number }[];   // hasta 6, del más viejo al más nuevo
}

export interface Attempt {
  id: string;
  hypothesis_id: string;
  stage: CycleStageId;
  ran_at: string;
  subject: string;                   // activo o cartera: 'Cartera de 11 sectores'
  snapshot_hash: string;             // 12 caracteres
  sharpe: number | null;             // null si el intento falló antes de calcular
  trades: number | null;
  pnl_pct: number | null;
  invalid: boolean;                  // marcado inválido por un bug: no cuenta
  invalid_reason: string | null;
}

export interface StageResult { stage: CycleStageId; verdict: Verdict; decided_at: string; reason: string; measure: Measure | null; imported: boolean; }

export interface Hypothesis {
  id: string;
  name: string;
  idea: string | null;               // NULO en la mayoría: la tarjeta no reserva espacio
  cycle: CycleId | null;             // null = historial importado
  stage: StageId;                    // etapa actual, donde cayó o fase posterior
  verdict: Verdict | null;           // de la última etapa evaluada; null en fase posterior
  status: HypothesisStatus;
  promoted: boolean;
  used_reserve: boolean;             // chip "Usó la reserva 2023+" (solo en la ficha)
  attempts_valid: number;
  attempts_invalid: number;
  last_metric: { name: string; value: number; at: string } | null;  // NULO en la mayoría
  discard_reason: string | null;     // con CORTA, ERROR o ALARMA
  discard_measure: Measure | null;
  updated_at: string;
}

export interface HypothesisDetail extends Hypothesis {
  timeline: StageResult[];           // solo etapas evaluadas; el cliente completa "no llegó"
  later_phases: { phase: LaterPhaseId; since: string; until: string | null; account_id: string | null }[];
  attempts: Attempt[];
  backtests: { id: string; label: string; stage: CycleStageId | null }[];  // vacío → "Sin backtests vinculados todavía"
  paper_account: { id: string; name: string; since: string } | null;
}

export interface Snapshot { name: string; as_of: string; hash: string; }   // hash de 12

/** Recibo inmutable guardado al cerrar el ciclo. Con ciclo abierto: mismo formato, partial=true. */
export interface CycleReceipt {
  id: string | null;                 // null si parcial
  hash: string | null;
  cycle: CycleId;
  partial: boolean;
  generated_at: string;
  counts: { entered: number; active: number; cut: number; held: number; promoted: number };
  stages: (StageCounts & { pending: boolean })[];
  drops: { hypothesis_id: string; name: string; stage: CycleStageId; verdict: Verdict; reason: string; measure: Measure | null }[];
  attempts: { hypothesis_id: string; name: string; valid: number; invalid: number }[];
  snapshots: Snapshot[];
}

/**
 * "Sin ciclo": historial importado. Son EVENTOS de etapa (E0–E3, `imported: true`), no hipótesis.
 * Los intentos importados tienen la etapa en texto libre ("comparar", …): no se mapean a E0–E3.
 */
export interface HistorySummary {
  stage_events: number;                                  // 55
  attempts: number;                                      // 202
  events_by_stage: { stage: CycleStageId; total: number; pass: number; cut: number; error: number; alarm: number }[];
  attempts_by_label: { label: string | null; count: number }[];   // label null → "(sin etiqueta)"
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
  source: 'alpaca' | 'strategy' | 'fred' | 'manual' | 'system';
  account_id: string | null;
}

/* ---------- Rendimiento (v0.3) — ver RENDIMIENTO.md ---------- */
export type PerfSubjectType = 'backtest' | 'run' | 'portfolio';
export type PerfRange = 'ytd' | '1y' | '3y' | '5y' | 'all';

export type MetricKey =
  | 'total_return' | 'cagr' | 'positive_months' | 'best_month' | 'worst_month'
  | 'volatility' | 'sharpe' | 'sortino' | 'max_drawdown' | 'calmar'
  | 'alpha' | 'beta' | 'relative_return';

export interface MetricValue {
  /** % como número (12.5 = 12,5 %), salvo cocientes (sharpe, sortino, calmar, beta). alpha en pp. */
  value: number | null;
  /** Lo decide el backend. 'not_computable' ⇒ value = null. */
  status: 'ok' | 'short_sample' | 'not_computable';
  /** Texto corto para la web: "Necesita 12 meses; hay 3,6". */
  reason: string | null;
  detail?: {
    peak_date?: string; trough_date?: string; recovery_date?: string | null; days?: number; // max_drawdown
    month?: string;                                                                       // best/worst_month 'YYYY-MM'
    positive?: number; total?: number;                                                    // positive_months
  };
}

export interface PerfSubject {
  type: PerfSubjectType;
  id: string;
  name: string;
  env: Env | null;                       // null en backtest
  run_label: string | null;              // '#bt-0412'
  account_id: string | null;             // run: cuenta donde corre; portfolio: la cuenta
  hypothesis_id: string | null;
  backtest_id: string | null;            // run: backtest original
  out_of_sample_start: string | null;    // backtest: inicio de la reserva
  costs_bps: number | null;
  strategies: { run_id: string; name: string; total_return: number }[] | null; // portfolio
}

export interface PerformanceReport {
  subject: PerfSubject;
  benchmark: { symbol: 'SPY' } | null;
  status: 'ready' | 'insufficient_data' | 'no_runs' | 'returns_not_computable';
  /**
   * Por qué no se puede calcular el retorno:
   * - 'unregistered_flows': cartera con aportes/retiros sin registrar (con registro: alguno sin conciliar, ver flows.unreconciled).
   * - 'capital_not_isolated': estrategia en ejecución sin capital propio; comparte posiciones y caja con la cuenta.
   */
  not_computable_reason: 'unregistered_flows' | 'capital_not_isolated' | null;
  range: PerfRange;
  available_ranges: { range: PerfRange; available: boolean; reason: string | null }[];
  period: { start: string; end: string; trading_days: number; complete_months: number; rebalances: number | null };
  as_of: string;
  stale: boolean;                        // siempre false en backtest
  aligned_with: string | null;           // 'backtest:bt-0415' si se pidió período común
  sample: {
    short: boolean;
    reasons: ('few_days' | 'few_months' | 'few_rebalances')[];
    not_computable: MetricKey[];
    unreliable: MetricKey[];
    months_until_full: number | null;
    /** Parámetros vigentes del backend (provisorios): 21 / 63 / 252 días y 12 rebalanceos. */
    thresholds: { descriptive_days: number; risk_days: number; annualized_days: number; min_rebalances: number };
  };
  /** Cartera: estado del registro de flujos. null en backtest. */
  flows: { registered: boolean; unreconciled: { date: string; amount: number }[] } | null;
  insufficient: { days_have: number; days_needed: number; first_report_estimate: string | null } | null;
  no_runs: { approved_at: string; account_id: string; next_rebalance_at: string | null } | null;
  headline: {
    total_return: number; benchmark_total_return: number | null;
    max_drawdown: number; max_dd_trough: string; max_dd_recovery_days: number | null;
    shadow_gap_pp: number | null;
  } | null;
  metrics: Record<MetricKey, MetricValue>;
  benchmark_metrics: Record<MetricKey, MetricValue>;
  series: {
    equity: { date: string; value: number; benchmark: number | null; shadow: number | null }[];
    drawdown: { date: string; dd: number; benchmark_dd: number | null }[];
  };
  rolling: {
    window_days: 126 | 252;
    points: { date: string; sharpe: number | null; volatility: number | null; benchmark_sharpe: number | null; benchmark_volatility: number | null }[];
  }[];
  monthly_returns: { year: number; month: number; ret: number; benchmark_ret: number | null; complete: boolean }[];
  annual_returns: { year: number; ret: number; benchmark_ret: number | null; partial: boolean }[];
  distributions: {
    monthly: ReturnDistribution;
    /** Vista secundaria: null con menos de `risk_days` días hábiles. */
    daily: (ReturnDistribution & { var_95: number; cvar_95: number }) | null;
    daily_reason: string | null;        // "Disponible desde 63 días; hay 41"
  };
  worst_drawdowns: {
    rank: number; peak_date: string; trough_date: string; recovery_date: string | null;
    depth: number; days_to_trough: number; days_to_recovery: number | null; total_days: number;
  }[];
  cash_flows: { date: string; amount: number }[] | null;   // portfolio
  conventions: {
    frequency: 'daily'; method: 'time_weighted'; annualization_days: 252;
    risk_free_source: string; risk_free_avg: number;
    benchmark: string; data_source: string; costs_bps: number | null; manual_url: string;
  };
}

export interface ReturnDistribution {
  bin_width_pct: number;
  bins: { from: number; to: number; count: number; benchmark_count: number | null }[];
  mean: number; benchmark_mean: number | null; n: number;
}

/** hypothesis_name null = backtest corrido suelto, sin hipótesis. */
export interface BacktestListItem {
  id: string; label: string; hypothesis_name: string | null; start: string; end: string; total_return: number;
}

export interface MarketClock {
  is_open: boolean;
  next_open: string;
  next_close: string;
}

/* ---------- Mercado (v0.4) — todo diario, al cierre de la sesión anterior ---------- */
export interface MarketAsOf {
  session_date: string;              // '2026-11-02' (cierre)
  updated_at: string;                // cuándo se guardó el snapshot (se muestra en ART)
  stale: boolean;
  missing_sessions: string[];        // sesiones sin snapshot (para el aviso)
}

export type TrendSide = 'above' | 'below';      // precio vs media de 200 días
export type VolLevel = 'low' | 'high';          // vol. 20 días vs su mediana de 1 año
export type RegimeId = 'up_calm' | 'up_vol' | 'down_calm' | 'down_vol';

export interface Regime {
  symbol: 'SPY';
  regime: RegimeId;
  since: string;
  sessions: number;
  price: number; ma200: number | null;           // null si hay menos de 200 sesiones
  vol20: number | null; vol_median_1y: number | null;
  strip: { regime: RegimeId | null; from: string; to: string; sessions: number }[];  // null = sin datos
}

export interface MotorState {
  state: 'risk_on' | 'neutral' | 'risk_off';
  computed_at: string;
  next_at: string;
  /** Texto del backend que explica el estado actual. null → solo el nombre del estado. */
  state_description: string | null;
}
/** null = el motor todavía no calculó (estado vacío). */
export type MotorStateResponse = MotorState | null;

export type ReturnKey = 'd1' | 'w1' | 'm1' | 'm3' | 'ytd';
/** null = historia insuficiente para ese plazo. Nunca 0. */
export type Returns = Record<ReturnKey, number | null>;

export interface SectorRow {
  symbol: string;                    // XLB … XLY, o SPY/EEM/EWZ
  name: string;
  is_reference: boolean;             // SPY, EEM, EWZ: sin ranking
  /** De las POSICIONES reales de cada cuenta (M4, M5) al cierre; nunca se deduce del ranking. Vacío = sin posición. */
  held_in: { account_id: string; account_name: string }[];
  returns: Returns;
  momentum_score: number | null;     // null si el motor no calculó
  rank: number | null;
  enabled: boolean;                  // top 3 con score > 0 entre los 11 (sean cuales sean)
  blocked: boolean;
  blocked_reason: string | null;     // 'score negativo', …
  score_as_of: string | null;
}

export interface MacroSeries {
  code: string;                      // 'T10Y2Y'
  title: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  status: 'active' | 'coming_soon';
  unit: 'pp' | 'pct' | 'index' | 'usd';
  last: { value: number; observation_date: string; published_at: string } | null;
  change_1m: number | null;
  /** Vencido = pasó expected_release + marketConfig.macroStale.graceBusinessDays sin dato nuevo;
   *  sin calendario FRED: 1,5 × frecuencia desde la última observación. */
  stale: boolean;
  expected_release: string | null;   // próxima publicación según el calendario de FRED; null = sin calendario
  points: { date: string; value: number }[];
}
export interface RecessionPeriod { start: string; end: string; }   // USREC

/** Sin 'cash': la caja de la cuenta no es parte del universo (lo más parecido es SHY). */
export type AssetClass = 'us_equity' | 'sector' | 'international' | 'bonds' | 'commodities' | 'currency' | 'real_estate';
export interface EtfRow {
  symbol: string;
  name: string;
  asset_class: AssetClass;
  close: number | null;
  returns: Returns;
  history_sessions: number;          // para explicar "sin datos"
  in_universe_since: string;
  spark_30d: number[] | null;        // null si < 2 puntos
  held_in: { account_id: string; account_name: string }[];   // vacío = no está en cartera
}
export interface DailyCandle { date: string; open: number; high: number; low: number; close: number; }

export interface MarketPage {
  as_of: MarketAsOf;
  clock: MarketClock;
  regime: Regime | null;
  motor: MotorStateResponse;
  sectors: SectorRow[];
  macro: MacroSeries[];
  recessions: RecessionPeriod[];
  etfs: EtfRow[];
  events: CalendarEvent[];           // filtrado: macro | fomc | holiday | early_close, 14 días
}
