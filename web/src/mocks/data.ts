import type { AccountSummary, MarketClock, Order, Position } from '../api/types'

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

export const positions = [
  { symbol: 'SPY', description: 'ETF S&P 500', side: 'long', qty: 20, avg_entry_price: 490, current_price: 500, market_value: 10000, unrealized_pl: 200, unrealized_pl_pct: 0.02, stop_price: 475, opened_at: '2026-10-01T14:30:00Z' },
  { symbol: 'TSLA', description: 'Tesla', side: 'short', qty: 10, avg_entry_price: 205, current_price: 200, market_value: 2000, unrealized_pl: 50, unrealized_pl_pct: 0.025, stop_price: 220, opened_at: '2026-10-15T14:30:00Z' },
] satisfies Position[]

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
