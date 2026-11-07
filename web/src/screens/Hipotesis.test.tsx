import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { Hipotesis } from './Hipotesis'
import { handlers } from '../mocks/handlers'
import { hypothesisDetail } from '../mocks/data'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { server.resetHandlers(); cleanup() })
afterAll(() => server.close())

describe('Hipotesis', () => {
  it('muestra idea, etapas, intentos y navega al ciclo', async () => {
    const navigate = vi.fn()
    server.use(http.get('*/api/hypotheses/:id', () => HttpResponse.json({ ...hypothesisDetail, idea: 'Idea de prueba' })))
    render(<Hipotesis id="h-1" navigate={navigate} />)
    expect(await screen.findByText(hypothesisDetail.name)).toBeTruthy()
    expect(screen.getByText('Idea de prueba')).toBeTruthy()
    expect(screen.getAllByText('No llegó')).toHaveLength(3)
    fireEvent.click(screen.getByRole('link', { name: hypothesisDetail.cycle! }))
    expect(navigate).toHaveBeenCalledWith(`/embudo/ciclo/${hypothesisDetail.cycle}`)
  })
  it('omite idea, muestra backtests vacíos e intento inválido en texto', async () => {
    server.use(http.get('*/api/hypotheses/:id', () => HttpResponse.json({ ...hypothesisDetail, idea: null, backtests: [], attempts: [{ id: 'a1', hypothesis_id: 'h-1', stage: 'e0', ran_at: '2026-10-30T00:00:00Z', subject: 'QQQ', snapshot_hash: 'abcd1234abcd', sharpe: null, trades: null, pnl_pct: null, invalid: true, invalid_reason: 'Datos duplicados' }] })))
    const { container } = render(<Hipotesis id="h-1" navigate={() => {}} />)
    expect(await screen.findByText('Sin backtests vinculados todavía')).toBeTruthy()
    expect(screen.getByText(/Inválido: Datos duplicados/)).toBeTruthy()
    expect(screen.getByText(/Sharpe — · Operaciones — · PnL —/)).toBeTruthy()
    expect(container.textContent).not.toContain('Idea de la hipótesis')
  })
  it('muestra 404 y permite reintentar errores', async () => {
    server.use(http.get('*/api/hypotheses/:id', () => new HttpResponse(null, { status: 404 })))
    const { rerender } = render(<Hipotesis id="missing" navigate={() => {}} />)
    expect(await screen.findByText('No existe la hipótesis.')).toBeTruthy()
    server.use(http.get('*/api/hypotheses/:id', () => HttpResponse.error()))
    rerender(<Hipotesis id="h-1" navigate={() => {}} />)
    expect(await screen.findByText('No pudimos cargar la hipótesis.')).toBeTruthy()
    server.use(http.get('*/api/hypotheses/:id', () => HttpResponse.json(hypothesisDetail)))
    fireEvent.click(screen.getByRole('button', { name: 'Reintentar' }))
    expect(await screen.findByText(hypothesisDetail.name)).toBeTruthy()
  })
})
