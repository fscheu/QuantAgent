import type { AccountRef, AccountSummary, CalendarEvent, CycleReceipt, CycleSummary, DailyCandle, Dividend, EquityPoint, FeedEvent, Funnel, HistorySummary, HomeStatus, Hypothesis, HypothesisDetail, MarketClock, MarketPage, Order, Position, ReviewRule, SectorExposure } from '../api/types'

const asOf = '2026-10-30T19:30:00Z'
export const etfCandles: DailyCandle[] = [
  { date: '2026-10-29', open: 100, high: 103, low: 99, close: 102 },
  { date: '2026-10-30', open: 102, high: 104, low: 101, close: 103 },
]

export const accounts = [
  {
    id: 'paper-m4', name: 'Paper M4', env: 'paper', status: 'active', as_of: asOf, stale: false,
    equity: 13000, long_value: 10000, short_value: 2000, cash: 5000,
    day_change: 125, day_change_pct: 0.0097, week_change: 310, week_change_pct: 0.0244,
    positions_count: 2, orders_today: 5, orders_rejected_today: 1,
    alarms: [{ id: 'alarm-1', account_id: 'paper-m4', severity: 'warning', kind: 'shadow_divergence', title: 'Sombra desalineada', message: 'El broker y la simulación local difieren 0,8 %.', since: asOf }],
    spark_30d: [12400, 12600, 12550, 12875, 13000],
  },
  {
    id: 'paper-m3', name: 'Paper M3', env: 'paper', status: 'active', as_of: asOf, stale: false,
    equity: 25000, long_value: 18000, short_value: 0, cash: 7000,
    day_change: -80, day_change_pct: -0.0032, week_change: 220, week_change_pct: 0.0089,
    positions_count: 1, orders_today: 2, orders_rejected_today: 0, alarms: [], spark_30d: null,
  },
  {
    id: 'live-main', name: 'Live', env: 'live', status: 'no_data', no_data_reason: 'not_funded', checked_at: asOf, next_check_at: null,
  },
] satisfies AccountSummary[]
export const accountRefs = accounts.map(({ id, name, env }) => ({ id, name, env })) satisfies AccountRef[]
export const homeStatus = { level: 'warning', headline: 'Casi todo bien: 1 advertencia', message: 'El broker y la simulación local difieren 0,8 %.', link: '/cuenta/paper-m4' } satisfies HomeStatus
export const feedEvents = [
  { id: 'drawdown-1', type: 'drawdown_limit', severity: 'critical', occurred_at: '2026-10-30T19:20:00Z', account_id: 'paper-m4', title: 'Drawdown sobre el límite en Paper M4', subtitle: '−8,4 % desde el máximo; límite 8 %', link: '/cuenta/paper-m4', group_count: 1 },
  { id: 'reject-1', type: 'order_rejected', severity: 'warning', occurred_at: '2026-10-30T18:00:00Z', account_id: 'paper-m4', title: '2 órdenes rechazadas en Paper M4', subtitle: 'Poder de compra insuficiente', link: '/cuenta/paper-m4?tab=ordenes', group_count: 2 },
  { id: 'cycle-1', type: 'cycle_closed', severity: 'info', occurred_at: '2026-10-30T17:00:00Z', account_id: null, title: 'Terminó el ciclo 12 del embudo', subtitle: '24 entraron, 3 siguen, 21 cayeron', link: '/embudo?cycle=12', group_count: 1 },
  { id: 'filled-1', type: 'order_filled', severity: 'neutral', occurred_at: '2026-10-30T16:00:00Z', account_id: 'paper-m3', title: '1 compra ejecutada', subtitle: 'MSFT 25 a 431,55', link: '/cuenta/paper-m3?tab=ordenes', group_count: 1 },
] satisfies FeedEvent[]
export const calendarEvents = [
  { id: 'calendar-2', kind: 'macro', starts_at: '2026-11-05T13:30:00Z', date: '2026-11-05', title: 'Empleo', subtitle: null, source: 'fred', account_id: null },
  { id: 'calendar-1', kind: 'rebalance', starts_at: null, date: '2026-11-02', title: 'Rebalanceo', subtitle: 'Paper M4', source: 'strategy', account_id: 'paper-m4' },
] satisfies CalendarEvent[]

export const positions = [
  { symbol: 'SPY', description: 'ETF S&P 500', side: 'long', qty: 20, status: 'executed', avg_entry_price: 490, current_price: 500, market_value: 10000, unrealized_pl: 200, unrealized_pl_pct: 0.02, stop_price: 475, opened_at: '2026-10-01T14:30:00Z' },
  { symbol: 'TSLA', description: 'Tesla', side: 'short', qty: 10, status: 'executed', avg_entry_price: 205, current_price: 200, market_value: 2000, unrealized_pl: 50, unrealized_pl_pct: 0.025, stop_price: 220, opened_at: '2026-10-15T14:30:00Z' },
] satisfies Position[]
export const equity: EquityPoint[] = [
  { date: '2026-10-26', account: 12650, benchmark: 681, shadow: 12620 },
  { date: '2026-10-27', account: 12810, benchmark: 686, shadow: 12770 },
  { date: '2026-10-28', account: 12760, benchmark: 684, shadow: 12740 },
  { date: '2026-10-29', account: 12920, benchmark: 690, shadow: 12880 },
  { date: '2026-10-30', account: 13000, benchmark: 693, shadow: 12900 },
]
export const dividends = [{ date: '2026-10-15', symbols: ['SPY', 'MSFT'], net: 24.8, withheld: 3.2 }] satisfies Dividend[]
export const sectorExposure = [
  { sector: 'Tecnología', long: 4200, short: 800, pct: 0.5 },
  { sector: 'Consumo', long: 1800, short: 0, pct: 0.18 },
] satisfies SectorExposure[]

export const orders = [
  { id: 'order-filled', symbol: 'SPY', side: 'buy', type: 'limit', qty: 2, limit_price: 500, stop_price: null, status: 'filled', filled_qty: 2, filled_avg_price: 499.5, time_in_force: 'day', submitted_at: asOf, cancelled_at: null, expired_at: null, reject_reason: null, reject_detail: null, reject_reason_raw: null },
  { id: 'order-pending', symbol: 'NVDA', side: 'buy', type: 'limit', qty: 1, limit_price: 140, stop_price: null, status: 'pending', filled_qty: 0, filled_avg_price: null, time_in_force: 'gtc', submitted_at: asOf, cancelled_at: null, expired_at: null, reject_reason: null, reject_detail: null, reject_reason_raw: null },
  { id: 'order-rejected', symbol: 'AAPL', side: 'buy', type: 'market', qty: 100, limit_price: null, stop_price: null, status: 'rejected', filled_qty: 0, filled_avg_price: null, time_in_force: 'day', submitted_at: asOf, cancelled_at: null, expired_at: null, reject_reason: 'Poder de compra insuficiente', reject_detail: 'Necesitaba USD 20.000; había USD 5.000.', reject_reason_raw: 'insufficient buying power' },
  { id: 'order-cancelled', symbol: 'AMD', side: 'sell', type: 'limit', qty: 1, limit_price: 180, stop_price: null, status: 'cancelled', filled_qty: 0, filled_avg_price: null, time_in_force: 'day', submitted_at: asOf, cancelled_at: asOf, expired_at: null, reject_reason: null, reject_detail: null, reject_reason_raw: null },
  { id: 'order-expired', symbol: 'MSFT', side: 'buy', type: 'limit', qty: 1, limit_price: 400, stop_price: null, status: 'expired', filled_qty: 0, filled_avg_price: null, time_in_force: 'day', submitted_at: asOf, cancelled_at: null, expired_at: asOf, reject_reason: null, reject_detail: null, reject_reason_raw: null },
] satisfies Order[]

export const marketClock = {
  is_open: false,
  next_open: '2026-11-02T14:30:00Z',
  next_close: '2026-11-02T21:00:00Z',
} satisfies MarketClock

// Valores numéricos de ejemplo; universo y estados siguen el canvas de Mercado.
const marketReturns = { d1: 0.4, w1: 1.2, m1: 2.8, m3: 5.1, ytd: 12.4 }
const etfNames = ['SPY|S&P 500|us_equity', 'QQQ|Nasdaq 100|us_equity', 'IWM|Russell 2000|us_equity', 'XLB|Materiales|sector', 'XLE|Energía|sector', 'XLF|Finanzas|sector', 'XLI|Industriales|sector', 'XLK|Tecnología|sector', 'XLP|Consumo básico|sector', 'XLU|Servicios públicos|sector', 'XLV|Salud|sector', 'XLY|Consumo discrecional|sector', 'EFA|Desarrollados|international', 'EEM|Emergentes|international', 'EWZ|Brasil|international', 'VGK|Europa|international', 'VWO|Emergentes amplios|international', 'AGG|Bonos agregados|bonds', 'BIL|Letras cortas|bonds', 'IEF|Treasuries 7-10 años|bonds', 'LQD|Bonos corporativos|bonds', 'SHY|Treasuries 1-3 años|bonds', 'TLT|Treasuries 20+ años|bonds', 'DBC|Materias primas|commodities', 'GLD|Oro|commodities', 'PDBC|Materias primas diversificadas|commodities', 'USO|Petróleo|commodities', 'UUP|Dólar|currency', 'VNQ|Inmobiliario EE.UU.|real_estate']
const marketEtfs = etfNames.map((item, index) => { const [symbol, name, asset_class] = item.split('|'); return { symbol, name, asset_class: asset_class as 'us_equity' | 'sector' | 'international' | 'bonds' | 'commodities' | 'currency' | 'real_estate', close: 100 + index, returns: { ...marketReturns, m1: index === 0 ? null : marketReturns.m1 }, history_sessions: index === 0 ? 13 : 252, in_universe_since: '2020-01-02', spark_30d: index === 0 ? null : [99, 100, 101], held_in: index === 6 ? [{ account_id: 'paper-m4', account_name: 'Paper M4' }] : [] } })
const sectorNames = ['XLB|Materiales', 'XLE|Energía', 'XLF|Finanzas', 'XLI|Industriales', 'XLK|Tecnología', 'XLP|Consumo básico', 'XLU|Servicios públicos', 'XLV|Salud', 'XLY|Consumo discrecional', 'SPY|S&P 500', 'EEM|Emergentes', 'EWZ|Brasil']
export const marketPage = {
  as_of: { session_date: '2026-10-30', updated_at: asOf, stale: false, missing_sessions: [] }, clock: marketClock,
  regime: { symbol: 'SPY', regime: 'up_calm', since: '2026-08-03', sessions: 62, price: 500, ma200: 480, vol20: 0.12, vol_median_1y: 0.16, strip: [{ regime: 'up_calm', from: '2025-11-03', to: '2026-10-30', sessions: 252 }, { regime: null, from: '2026-10-31', to: '2026-10-31', sessions: 1 }] },
  motor: null,
  sectors: sectorNames.map((item, index) => { const [symbol, name] = item.split('|'); return { symbol, name, is_reference: index >= 9, held_in: [], returns: marketReturns, momentum_score: index < 9 ? 1 - index / 8 : null, rank: index < 9 ? index + 1 : null, enabled: index < 3, blocked: index >= 3 && index < 9, blocked_reason: index >= 3 && index < 9 ? 'score negativo' : null, score_as_of: index < 9 ? '2026-10-30' : null } }),
  macro: ['T10Y2Y|Treasury 10 años menos 2 años|daily|pp|active', 'FEDFUNDS|Tasa de fondos federales|monthly|pct|coming_soon', 'CPIAUCSL|IPC|monthly|index|coming_soon', 'UNRATE|Desempleo|monthly|pct|coming_soon', 'DGS10|Treasury 10 años|daily|pct|coming_soon', 'DGS2|Treasury 2 años|daily|pct|coming_soon', 'T10YIE|Inflación esperada a 10 años|daily|pct|coming_soon', 'VIXCLS|Volatilidad implícita|daily|index|coming_soon', 'PAYEMS|Empleo no agrícola|monthly|index|coming_soon', 'INDPRO|Producción industrial|monthly|index|coming_soon'].map((item) => { const [code, title, frequency, unit, status] = item.split('|'); const active = status === 'active'; return { code, title, frequency: frequency as 'daily' | 'weekly' | 'monthly', unit: unit as 'pp' | 'pct' | 'index' | 'usd', status: status as 'active' | 'coming_soon', last: active ? { value: 0.12, observation_date: '2026-10-30', published_at: asOf } : null, change_1m: active ? -0.08 : null, stale: false, expected_release: null, points: active ? [{ date: '2026-10-01', value: 0.2 }, { date: '2026-10-30', value: 0.12 }] : [] } }),
  recessions: [{ start: '2020-02-01', end: '2020-04-01' }], etfs: marketEtfs, events: calendarEvents,
} satisfies MarketPage

// Números de ejemplo del canvas: 24 hipótesis, 3 promovidas; historial 55/202.
export const funnel = { cycle: 'all', period: 'all', total: 24, stages: [{ stage: 'e0', reached: 24, here: 0, passed: 18, cut: 4, error: 1, alarm: 1 }, { stage: 'e1', reached: 18, here: 0, passed: 12, cut: 4, error: 1, alarm: 1 }, { stage: 'e2', reached: 12, here: 0, passed: 7, cut: 3, error: 1, alarm: 1 }, { stage: 'e3', reached: 7, here: 0, passed: 3, cut: 3, error: 0, alarm: 1 }], later: [{ phase: 'shadow', reached: 2, here: 2 }, { phase: 'paper', reached: 1, here: 1 }, { phase: 'candidate', reached: 0, here: 0 }], promoted: 3, drop_reasons: [{ stage: 'e1', verdict: 'CORTA', cause_code: 'sharpe_below', cause_label: 'Sharpe bajo el umbral', count: 4, example: { metric: 'Sharpe', value: 0.31, threshold: 0.5, unit: 'ratio', comparator: '>=' } }], sample_too_small: false } satisfies Funnel
export const currentCycle = { cycle: '2026-11', status: 'open', started_at: '2026-11-01T00:00:00Z', closed_at: null, entered: 3, active: 2, cut: 0, held: 1, promoted: 0, reached_by_stage: { e0: 3, e1: 1, e2: null, e3: null }, milestones: [{ key: 'proposal', day: 1, date: '2026-11-01', done: true }, { key: 'funnel_start', day: null, date: '', done: false }, { key: 'close', day: null, date: '', done: false }], headline: null, receipt_id: null } satisfies CycleSummary
export const cycles = [currentCycle, { ...currentCycle, cycle: '2026-10', status: 'closed', closed_at: '2026-10-31T23:59:59Z', entered: 24, active: 2, cut: 18, held: 1, promoted: 3, reached_by_stage: { e0: 24, e1: 18, e2: 12, e3: 7 }, headline: '24 entraron, 3 se promovieron', receipt_id: 'receipt-2026-10' }] satisfies CycleSummary[]
export const cycleReceipts: CycleReceipt[] = [
  { id: 'receipt-2026-10', hash: '0123456789abcdef', cycle: '2026-10', partial: false, generated_at: asOf, counts: { entered: 24, active: 2, cut: 18, held: 1, promoted: 3 }, stages: [{ stage: 'e0', reached: 24, here: 24, passed: 18, cut: 4, error: 1, alarm: 1, pending: false }], drops: [{ hypothesis_id: 'h1', name: 'Momentum sectorial', stage: 'e0', verdict: 'CORTA', reason: 'Sharpe bajo', measure: { metric: 'Sharpe', value: 0.31, threshold: 0.5, unit: 'ratio', comparator: '>=' } }], attempts: [{ hypothesis_id: 'h1', name: 'Momentum sectorial', valid: 3, invalid: 1 }], snapshots: [{ name: 'mercado', as_of: asOf, hash: 'abcdef012345' }] },
  { id: null, hash: null, cycle: '2026-11', partial: true, generated_at: asOf, counts: { entered: 3, active: 2, cut: 0, held: 1, promoted: 0 }, stages: [{ stage: 'e0', reached: 3, here: 3, passed: 2, cut: 0, error: 0, alarm: 1, pending: false }, { stage: 'e1', reached: 1, here: 1, passed: 1, cut: 0, error: 0, alarm: 0, pending: true }], drops: [], attempts: [], snapshots: [{ name: 'mercado', as_of: '', hash: 'fedcba987654' }] },
]
export const reviewRule = { limit: 6, streak: 2, reached: false, last_cycles: [{ cycle: '2026-09', promoted: 0 }, { cycle: '2026-10', promoted: 3 }] } satisfies ReviewRule
export const history = { stage_events: 55, attempts: 202, events_by_stage: [{ stage: 'e0', total: 20, pass: 10, cut: 6, error: 2, alarm: 2 }, { stage: 'e1', total: 15, pass: 8, cut: 5, error: 1, alarm: 1 }, { stage: 'e2', total: 12, pass: 6, cut: 4, error: 1, alarm: 1 }, { stage: 'e3', total: 8, pass: 3, cut: 4, error: 0, alarm: 1 }], attempts_by_label: [{ label: 'comparar', count: 150 }, { label: null, count: 52 }] } satisfies HistorySummary
export const hypotheses = [{ id: 'h1', name: 'Momentum sectorial', idea: null, cycle: '2026-10', stage: 'shadow', verdict: null, status: 'promoted', promoted: true, used_reserve: false, attempts_valid: 4, attempts_invalid: 0, last_metric: null, discard_reason: null, discard_measure: null, updated_at: asOf }] satisfies Hypothesis[]
export const hypothesisDetail = { ...hypotheses[0], timeline: [{ stage: 'e3', verdict: 'PASA', decided_at: asOf, reason: 'Superó el umbral', measure: null, imported: false }], later_phases: [{ phase: 'shadow', since: asOf, until: null, account_id: null }], attempts: [], backtests: [{ id: '41', label: 'Momentum sectorial', stage: 'e3' }], paper_account: null } satisfies HypothesisDetail

import type { BacktestListItem, PerformanceReport } from '../api/types'
export const backtests: BacktestListItem[] = [
 { id:'41', label:'Rotación sectorial por momentum 3M', hypothesis_name:'Momentum sectorial', start:'2018-01-02', end:'2026-10-30', total_return:84.2 },
 { id:'42', label:'Reversión a la media en ETFs', hypothesis_name:null, start:'2020-01-02', end:'2026-10-30', total_return:46.8 },
]
const metric = (value:number|null,status:'ok'|'short_sample'|'not_computable'='ok',reason:string|null=null) => ({value,status,reason})
export const performance: PerformanceReport = {
 subject:{type:'backtest',id:'41',name:backtests[0].label,env:null,run_label:'#41',account_id:null,hypothesis_id:'h1',backtest_id:'41',out_of_sample_start:null,costs_bps:5,strategies:null}, benchmark:{symbol:'SPY'},status:'ready',not_computable_reason:null,range:'all',available_ranges:[{range:'ytd',available:true,reason:null},{range:'1y',available:true,reason:null},{range:'3y',available:true,reason:null},{range:'5y',available:true,reason:null},{range:'all',available:true,reason:null}],period:{start:'2018-01-02',end:'2026-10-30',trading_days:2200,complete_months:105,rebalances:42},as_of:'2026-10-30',stale:false,aligned_with:null,
 sample:{short:true,reasons:['few_months'],not_computable:['cagr','calmar','alpha'],unreliable:['sharpe'],months_until_full:2,thresholds:{descriptive_days:21,risk_days:63,annualized_days:252,min_rebalances:12}},flows:null,insufficient:null,no_runs:null,
 headline:{total_return:84.2,benchmark_total_return:null,max_drawdown:-18.3,max_dd_trough:'2022-06-30',max_dd_recovery_days:14,shadow_gap_pp:null},
 metrics:{total_return:metric(84.2),cagr:metric(null,'not_computable','Necesita 12 meses; hay 3,6'),positive_months:metric(62),best_month:metric(8.4),worst_month:metric(-7.2),volatility:metric(14.2),sharpe:metric(1.12,'short_sample','Muestra corta'),sortino:metric(1.4),max_drawdown:metric(-18.3),calmar:metric(null,'not_computable','Necesita 12 meses; hay 3,6'),alpha:metric(null,'not_computable','Benchmark SPY no disponible'),beta:metric(null,'not_computable','Benchmark SPY no disponible'),relative_return:metric(null,'not_computable','Benchmark SPY no disponible')},
 benchmark_metrics:{total_return:metric(null,'not_computable','Benchmark SPY no disponible'),cagr:metric(null,'not_computable','Benchmark SPY no disponible'),positive_months:metric(null,'not_computable','Benchmark SPY no disponible'),best_month:metric(null,'not_computable','Benchmark SPY no disponible'),worst_month:metric(null,'not_computable','Benchmark SPY no disponible'),volatility:metric(null,'not_computable','Benchmark SPY no disponible'),sharpe:metric(null,'not_computable','Benchmark SPY no disponible'),sortino:metric(null,'not_computable','Benchmark SPY no disponible'),max_drawdown:metric(null,'not_computable','Benchmark SPY no disponible'),calmar:metric(null,'not_computable','Benchmark SPY no disponible'),alpha:metric(null,'not_computable','Benchmark SPY no disponible'),beta:metric(null,'not_computable','Benchmark SPY no disponible'),relative_return:metric(null,'not_computable','Benchmark SPY no disponible')},
 series:{equity:[{date:'2026-01-01',value:10000,benchmark:null,shadow:null},{date:'2026-02-01',value:10840,benchmark:null,shadow:null}],drawdown:[{date:'2026-01-01',dd:0,benchmark_dd:null},{date:'2026-02-01',dd:-2.1,benchmark_dd:null}]},rolling:[{window_days:126,points:[{date:'2026-02-01',sharpe:1.12,volatility:14.2,benchmark_sharpe:null,benchmark_volatility:null}]}],monthly_returns:[{year:2026,month:1,ret:8.4,benchmark_ret:null,complete:true}],annual_returns:[{year:2026,ret:8.4,benchmark_ret:null,partial:true}],distributions:{monthly:{bin_width_pct:2,bins:[{from:-2,to:0,count:2,benchmark_count:null},{from:0,to:2,count:4,benchmark_count:null}],mean:1.2,benchmark_mean:null,n:6},daily:null,daily_reason:'Disponible desde 63 días; hay 20'},worst_drawdowns:[{rank:1,peak_date:'2026-05-01',trough_date:'2026-06-01',recovery_date:null,depth:-18.3,days_to_trough:22,days_to_recovery:null,total_days:30}],cash_flows:null,conventions:{frequency:'daily',method:'time_weighted',annualization_days:252,risk_free_source:'Treasury 3M',risk_free_avg:0.04,benchmark:'SPY con dividendos',data_source:'Motor de backtest',costs_bps:5,manual_url:'/manual'}
}
