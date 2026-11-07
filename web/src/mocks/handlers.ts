import { http, HttpResponse } from 'msw'
import type { AccountRef, AccountSummary, CalendarEvent, CycleSummary, DailyCandle, Dividend, EquityPoint, FeedEvent, Funnel, HistorySummary, HomeStatus, Hypothesis, HypothesisDetail, MarketPage, Order, Position, ReviewRule, SectorExposure } from '../api/types'
import { accountRefs, accounts, backtests, calendarEvents, currentCycle, cycles, dividends, equity, etfCandles, feedEvents, funnel, history, homeStatus, hypotheses, hypothesisDetail, marketClock, marketPage, orders, performance, positions, reviewRule, sectorExposure } from './data'

/** Forma que devuelve la API real (OrderResponse); el cliente la convierte a Order. */
const toOrderResponse = (o: Order) => ({
  id: Number(o.id.replace(/\D/g, '')) || 0, symbol: o.symbol, side: o.side, type: o.type, status: o.status, qty: o.qty,
  filled_qty: o.filled_qty, price: o.limit_price, stop_price: o.stop_price, avg_fill_price: o.filled_avg_price,
  created_at: o.submitted_at, filled_at: null, reject_reason: o.reject_reason, reject_detail: o.reject_detail,
  reject_reason_raw: o.reject_reason_raw,
})

export const handlers = [
  http.get('*/api/backtests', () => HttpResponse.json(backtests)),
  http.get('*/api/performance/backtest/:id', ({ params, request }) => { const align = new URL(request.url).searchParams.get('align_with'); return HttpResponse.json({ ...performance, subject: { ...performance.subject, id: String(params.id) }, aligned_with: align }) }),
  http.get('*/api/accounts', () => HttpResponse.json<AccountRef[]>(accountRefs)),
  http.get('*/api/accounts/:accountId/summary', ({ params }) => HttpResponse.json<AccountSummary>(accounts.find(({ id }) => id === params.accountId)!)),
  http.get('*/api/feed', () => HttpResponse.json<FeedEvent[]>(feedEvents)),
  http.get('*/api/calendar', () => HttpResponse.json<CalendarEvent[]>(calendarEvents)),
  http.get('*/api/home/status', () => HttpResponse.json<HomeStatus>(homeStatus)),
  http.get('*/api/accounts/:accountId/positions', ({ params }) =>
    HttpResponse.json<Position[]>(params.accountId === 'paper-m4' ? positions : [])),
  http.get('*/api/accounts/:accountId/equity', ({ params }) => HttpResponse.json<EquityPoint[]>(params.accountId === 'paper-m4' ? equity : [])),
  http.get('*/api/accounts/:accountId/dividends', ({ params }) => HttpResponse.json<Dividend[]>(params.accountId === 'paper-m4' ? dividends : [])),
  http.get('*/api/accounts/:accountId/sector-exposure', ({ params }) => HttpResponse.json<SectorExposure[]>(params.accountId === 'paper-m4' ? sectorExposure : [])),
  http.get('*/api/accounts/:accountId/orders', ({ params }) =>
    HttpResponse.json((params.accountId === 'paper-m4' ? orders : []).map(toOrderResponse))),
  http.get('*/api/market/clock', () => HttpResponse.json(marketClock)),
  http.get('*/api/market', () => HttpResponse.json<MarketPage>(marketPage)),
  http.get('*/api/market/series', ({ request }) => { const params = new URL(request.url).searchParams; return HttpResponse.json(params.get('symbols')?.split(',').map(symbol => ({ symbol, points: symbol === 'QQQ' ? [] : [{ date: '2026-10-01', value: 100 }, { date: '2026-10-30', value: 104 }] })) ?? []) }),
  http.get('*/api/market/etf/:symbol/candles', () => HttpResponse.json<DailyCandle[]>(etfCandles)),
  http.get('*/api/funnel', () => HttpResponse.json<Funnel>(funnel)),
  http.get('*/api/funnel/cycles', () => HttpResponse.json<CycleSummary[]>(cycles)),
  http.get('*/api/funnel/cycles/current', () => HttpResponse.json<CycleSummary>(currentCycle)),
  http.get('*/api/funnel/review', () => HttpResponse.json<ReviewRule>(reviewRule)),
  http.get('*/api/funnel/history', () => HttpResponse.json<HistorySummary>(history)),
  http.get('*/api/hypotheses', () => HttpResponse.json<Hypothesis[]>(hypotheses)),
  http.get('*/api/hypotheses/:id', () => HttpResponse.json<HypothesisDetail>(hypothesisDetail)),
]
