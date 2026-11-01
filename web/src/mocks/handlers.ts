import { http, HttpResponse } from 'msw'
import type { AccountRef, AccountSummary, CalendarEvent, FeedEvent, HomeStatus, Order, Position } from '../api/types'
import { accountRefs, accounts, calendarEvents, feedEvents, homeStatus, marketClock, orders, positions } from './data'

/** Forma que devuelve la API real (OrderResponse); el cliente la convierte a Order. */
const toOrderResponse = (o: Order) => ({
  id: Number(o.id.replace(/\D/g, '')) || 0, symbol: o.symbol, side: o.side, type: o.type, status: o.status, qty: o.qty,
  filled_qty: o.filled_qty, price: o.limit_price, stop_price: o.stop_price, avg_fill_price: o.filled_avg_price,
  created_at: o.submitted_at, filled_at: null, reject_reason: o.reject_reason, reject_detail: o.reject_detail,
  reject_reason_raw: o.reject_reason_raw,
})

export const handlers = [
  http.get('*/api/accounts', () => HttpResponse.json<AccountRef[]>(accountRefs)),
  http.get('*/api/accounts/:accountId/summary', ({ params }) => HttpResponse.json<AccountSummary>(accounts.find(({ id }) => id === params.accountId)!)),
  http.get('*/api/feed', () => HttpResponse.json<FeedEvent[]>(feedEvents)),
  http.get('*/api/calendar', () => HttpResponse.json<CalendarEvent[]>(calendarEvents)),
  http.get('*/api/home/status', () => HttpResponse.json<HomeStatus>(homeStatus)),
  http.get('*/api/accounts/:accountId/positions', ({ params }) =>
    HttpResponse.json<Position[]>(params.accountId === 'paper-m4' ? positions : [])),
  http.get('*/api/accounts/:accountId/orders', ({ params }) =>
    HttpResponse.json((params.accountId === 'paper-m4' ? orders : []).map(toOrderResponse))),
  http.get('*/api/market/clock', () => HttpResponse.json(marketClock)),
]
