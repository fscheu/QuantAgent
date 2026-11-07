import { cleanup, render, screen, waitFor, fireEvent } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { ThemeProvider } from '../ThemeContext'
import type { EtfRow } from '../api/types'
import { handlers } from '../mocks/handlers'
import { etfCandles } from '../mocks/data'

vi.mock('lightweight-charts', () => ({ createChart: () => ({ addCandlestickSeries: () => ({ setData: vi.fn() }), remove: vi.fn() }) }))
import { EtfDetail } from './EtfDetail'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { cleanup(); server.resetHandlers() })
afterAll(() => server.close())
const row = { symbol: 'SPY', name: 'S&P 500', asset_class: 'us_equity' as const, close: 103, returns: { d1: null, w1: null, m1: null, m3: null, ytd: null }, history_sessions: 2, in_universe_since: '2026-01-01', spark_30d: null, held_in: [{ account_id: 'paper', account_name: 'Paper' }] }
const detail = (symbol: string, etfRow?: EtfRow) => <ThemeProvider><EtfDetail symbol={symbol} row={etfRow} navigate={vi.fn()} /></ThemeProvider>

describe('EtfDetail', () => {
  it('muestra cierre, delta calculado, fecha y cartera; el rango actualiza la consulta', async () => {
    const requested: string[] = []
    server.use(http.get('*/api/market/etf/:symbol/candles', ({ request }) => { requested.push(new URL(request.url).searchParams.get('range') ?? ''); return HttpResponse.json(etfCandles) }))
    render(detail('SPY', row))
    expect((await screen.findAllByText('103,00')).length).toBeGreaterThan(0)
    expect(screen.getByLabelText('subió 1,00 dólares')).toBeTruthy()
    expect(screen.getByText('Último dato: 2026-10-30')).toBeTruthy()
    expect(screen.getByText('En cartera hoy · Paper')).toBeTruthy()
    expect(screen.getAllByText('—')).toHaveLength(4)
    expect(screen.getByText('SPY cerró en 103,00')).toBeTruthy()
    expect(screen.getAllByTitle('2 sesiones disponibles')).toHaveLength(4)
    fireEvent.click(screen.getByRole('button', { name: 'Log' }))
    expect(screen.getByRole('button', { name: 'Log' }).getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: '3M' }))
    await waitFor(() => expect(requested).toContain('3m'))
    fireEvent.click(screen.getByRole('button', { name: '1A' }))
    await waitFor(() => expect(requested).toContain('1y'))
  })

  it('explica historia insuficiente y presenta una tabla accesible', async () => {
    server.use(http.get('*/api/market/etf/:symbol/candles', () => HttpResponse.json([etfCandles[0]])))
    render(detail('SPY'))
    expect(await screen.findByText('Sin datos suficientes para el gráfico.')).toBeTruthy()
    expect(screen.getByRole('table', { name: /datos diarios/i })).toBeTruthy()
  })

  it('permite reintentar una respuesta fallida', async () => {
    let calls = 0
    server.use(http.get('*/api/market/etf/:symbol/candles', () => ++calls === 1 ? HttpResponse.error() : HttpResponse.json(etfCandles)))
    render(detail('SPY'))
    fireEvent.click(await screen.findByRole('button', { name: 'Reintentar' }))
    expect(await screen.findByText('Último dato: 2026-10-30')).toBeTruthy()
  })
})
