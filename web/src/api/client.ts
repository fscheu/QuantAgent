import type { AccountRef, AccountSummary, CalendarEvent, Dividend, EquityPoint, FeedEvent, HomeStatus, Order, Position, SectorExposure } from './types'

export class ApiError extends Error {
  constructor(readonly status: number, message: string) { super(message); this.name = 'ApiError' }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) throw new ApiError(response.status, `Error ${response.status}`)
  return response.json() as Promise<T>
}

export const getAccountRefs = () => get<AccountRef[]>('/api/accounts')
export const getAccounts = async () => Promise.all((await getAccountRefs()).map(({ id }) => getSummary(id)))
export const getSummary = (accountId: string) => get<AccountSummary>(`/api/accounts/${encodeURIComponent(accountId)}/summary`)
export const getFeed = () => get<FeedEvent[]>('/api/feed')
export const getCalendar = (days: number) => get<CalendarEvent[]>(`/api/calendar?days=${days}`)
export const getHomeStatus = () => get<HomeStatus>('/api/home/status')
export const getPositions = (accountId: string) => get<Position[]>(`/api/accounts/${encodeURIComponent(accountId)}/positions`)
export type EquityRange = '1W' | '1M' | '3M' | 'YTD' | 'ALL'
export const getEquity = (accountId: string, range: EquityRange) => get<EquityPoint[]>(`/api/accounts/${encodeURIComponent(accountId)}/equity?range=${range}`)
export const getDividends = (accountId: string) => get<Dividend[]>(`/api/accounts/${encodeURIComponent(accountId)}/dividends`)
export const getSectorExposure = (accountId: string) => get<SectorExposure[]>(`/api/accounts/${encodeURIComponent(accountId)}/sector-exposure`)
/** Orden tal como la devuelve la API (OrderResponse del contrato OpenAPI). */
interface OrderResponse {
  id: number; symbol: string; side: Order['side']; type: Order['type']; status: Order['status']
  qty: number; filled_qty: number; price: number | null; stop_price: number | null; avg_fill_price: number | null
  created_at: string; filled_at: string | null
  reject_reason: string | null; reject_detail: string | null; reject_reason_raw: string | null
}

const toOrder = (o: OrderResponse): Order => ({
  id: String(o.id), symbol: o.symbol, side: o.side, type: o.type, qty: o.qty,
  limit_price: o.type === 'limit' || o.type === 'stop_limit' ? o.price : null, stop_price: o.stop_price,
  status: o.status, filled_qty: o.filled_qty, filled_avg_price: o.avg_fill_price, time_in_force: null,
  submitted_at: o.created_at, cancelled_at: null, expired_at: null,
  reject_reason: o.reject_reason, reject_detail: o.reject_detail, reject_reason_raw: o.reject_reason_raw,
})

export const getOrders = (accountId: string, date: string) =>
  get<OrderResponse[]>(`/api/accounts/${encodeURIComponent(accountId)}/orders?date=${encodeURIComponent(date)}`).then(rows => rows.map(toOrder))

import type { BacktestListItem, PerformanceReport, PerfRange } from './types'
export const getBacktests = () => get<BacktestListItem[]>('/api/backtests')
export const getBacktestPerformance = (id: string, range: PerfRange, vs: string | null) => {
  const params = new URLSearchParams({ range }); if (vs?.startsWith('backtest:')) params.set('align_with', vs)
  return get<PerformanceReport>(`/api/performance/backtest/${encodeURIComponent(id)}?${params}`)
}
export const getAccountPerformance = (type: 'portfolio' | 'run', id: string, range: PerfRange) =>
  get<PerformanceReport>(`/api/performance/${type}/${encodeURIComponent(id)}?range=${range}`)
