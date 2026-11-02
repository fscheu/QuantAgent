import type { AccountRef, AccountSummary, CalendarEvent, Dividend, EquityPoint, FeedEvent, HomeStatus, MarketClock, Order, Position, SectorExposure } from '../api/types'

const asOf = '2026-10-30T19:30:00Z'

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
  { id: 'cycle-1', type: 'cycle_completed', severity: 'info', occurred_at: '2026-10-30T17:00:00Z', account_id: null, title: 'Terminó el ciclo 12 del embudo', subtitle: '24 entraron, 3 siguen, 21 cayeron', link: '/embudo?cycle=12', group_count: 1 },
  { id: 'filled-1', type: 'order_filled', severity: 'neutral', occurred_at: '2026-10-30T16:00:00Z', account_id: 'paper-m3', title: '1 compra ejecutada', subtitle: 'MSFT 25 a 431,55', link: '/cuenta/paper-m3?tab=ordenes', group_count: 1 },
] satisfies FeedEvent[]
export const calendarEvents = [
  { id: 'calendar-2', kind: 'macro', starts_at: '2026-11-05T13:30:00Z', date: '2026-11-05', title: 'Empleo', subtitle: null, source: 'fred', account_id: null },
  { id: 'calendar-1', kind: 'rebalance', starts_at: null, date: '2026-11-02', title: 'Rebalanceo', subtitle: 'Paper M4', source: 'strategy', account_id: 'paper-m4' },
] satisfies CalendarEvent[]

export const positions = [
  { symbol: 'SPY', description: 'ETF S&P 500', side: 'long', qty: 20, avg_entry_price: 490, current_price: 500, market_value: 10000, unrealized_pl: 200, unrealized_pl_pct: 0.02, stop_price: 475, opened_at: '2026-10-01T14:30:00Z' },
  { symbol: 'TSLA', description: 'Tesla', side: 'short', qty: 10, avg_entry_price: 205, current_price: 200, market_value: 2000, unrealized_pl: 50, unrealized_pl_pct: 0.025, stop_price: 220, opened_at: '2026-10-15T14:30:00Z' },
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
  { sector: 'Tecnología', long: 42, short: 8, pct: 34 },
  { sector: 'Consumo', long: 18, short: 0, pct: 18 },
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
