import { http, HttpResponse } from 'msw'
import type { AccountSummary, Order, Position } from '../api/types'
import { accounts, marketClock, orders, positions } from './data'

export const handlers = [
  http.get('*/api/accounts', () => HttpResponse.json<AccountSummary[]>(accounts)),
  http.get('*/api/accounts/:accountId/summary', ({ params }) => {
    const account = accounts.find(({ id }) => id === params.accountId)!
    return HttpResponse.json<AccountSummary>(account)
  }),
  http.get('*/api/accounts/:accountId/positions', ({ params }) =>
    HttpResponse.json<Position[]>(params.accountId === 'paper-m4' ? positions : [])),
  http.get('*/api/accounts/:accountId/orders', ({ params }) =>
    HttpResponse.json<Order[]>(params.accountId === 'paper-m4' ? orders : [])),
  http.get('*/api/market/clock', () => HttpResponse.json(marketClock)),
]
