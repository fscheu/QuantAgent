import { http, HttpResponse } from 'msw'
import type { AccountSummary, Order, Position } from '../api/types'
import { accounts, marketClock, orders, positions } from './data'

/** Forma que devuelve la API real (OrderResponse); el cliente la convierte a Order. */
const toOrderResponse = (o: Order) => ({
  id: Number(o.id.replace(/\D/g, '')) || 0, symbol: o.symbol, side: o.side, type: o.type, status: o.status, qty: o.qty,
  filled_qty: o.filled_qty, price: o.limit_price, stop_price: o.stop_price, avg_fill_price: o.filled_avg_price,
  created_at: o.submitted_at, filled_at: null, reject_reason: o.reject_reason, reject_detail: o.reject_detail,
  reject_reason_raw: o.reject_reason_raw,
})

export const handlers = [
  http.get('*/api/accounts', () => HttpResponse.json<AccountSummary[]>(accounts)),
  http.get('*/api/accounts/:accountId/summary', ({ params }) => {
    const account = accounts.find(({ id }) => id === params.accountId)!
    return HttpResponse.json<AccountSummary>(account)
  }),
  http.get('*/api/accounts/:accountId/positions', ({ params }) =>
    HttpResponse.json<Position[]>(params.accountId === 'paper-m4' ? positions : [])),
  http.get('*/api/accounts/:accountId/orders', ({ params }) =>
    HttpResponse.json((params.accountId === 'paper-m4' ? orders : []).map(toOrderResponse))),
  http.get('*/api/market/clock', () => HttpResponse.json(marketClock)),
]
