import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { Cuenta } from './Cuenta'

// Respuestas con la forma que devuelve la API real (docs/api/openapi.json), incluidos los null que hoy
// salen de las bases paper: sin stop, sin descripción, sin día/semana, sin sparkline.
const server = setupServer(
  http.get('*/api/accounts', () => HttpResponse.json([{ id: 'cartera-x', name: 'cartera-x', env: 'paper' }])),
  http.get('*/api/accounts/:id/summary', () => HttpResponse.json({
    id: 'cartera-x', name: 'cartera-x', env: 'paper', status: 'active', as_of: '2026-10-30T08:00:27Z', stale: false,
    equity: 99881.53, long_value: 7060.77, short_value: 542.35, cash: 84707.42, day_change: null, day_change_pct: null,
    week_change: null, week_change_pct: null, positions_count: 1, orders_today: 2, orders_rejected_today: 0, alarms: [], spark_30d: null })),
  http.get('*/api/accounts/:id/positions', () => HttpResponse.json([{
    symbol: 'DBC', description: null, side: 'short', qty: 1, avg_entry_price: 33.85, current_price: 33.85, market_value: 33.85,
    unrealized_pl: 0, unrealized_pl_pct: 0, stop_price: null, stop_distance: null, opened_at: '2026-10-28T08:00:25Z' }])),
  http.get('*/api/accounts/:id/orders', () => HttpResponse.json([
    { id: 28, symbol: 'EEM', side: 'sell', type: 'market', status: 'pending', qty: 7, filled_qty: 0, price: 62.72, stop_price: null,
      avg_fill_price: null, created_at: '2026-10-30T08:01:28Z', filled_at: null, reject_reason: null, reject_detail: null, reject_reason_raw: null },
    { id: 29, symbol: 'XLB', side: 'buy', type: 'limit', status: 'filled', qty: 1, filled_qty: 1, price: 52.5, stop_price: null,
      avg_fill_price: 52.4, created_at: '2026-10-30T08:01:30Z', filled_at: '2026-10-30T13:30:01Z', reject_reason: null, reject_detail: null, reject_reason_raw: null },
  ])),
)
beforeAll(() => server.listen())
afterEach(() => cleanup())
afterAll(() => server.close())

describe('Cuenta con respuestas de la API real', () => {
  it('dibuja posiciones y órdenes sin romper aunque falten stop, descripción y variaciones', async () => {
    render(<Cuenta accountId="cartera-x" navigate={() => {}} />)
    expect((await screen.findAllByText('EEM')).length).toBeGreaterThan(0)
    expect(screen.getAllByText('XLB').length).toBeGreaterThan(0)
    expect(screen.getAllByText('DBC').length).toBeGreaterThan(0)
    expect(screen.queryByText('Invalid time value')).toBeNull()
  })
})
