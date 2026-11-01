import type { AccountRef, AccountSummary, CalendarEvent, FeedEvent, HomeStatus, Order, Position } from './types'

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
