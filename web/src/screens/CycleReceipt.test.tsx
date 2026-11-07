import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { handlers } from '../mocks/handlers'
import { cycleReceipts } from '../mocks/data'
import { CycleReceiptScreen } from './CycleReceipt'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { server.resetHandlers(); cleanup() })
afterAll(() => server.close())

describe('CycleReceiptScreen', () => {
  it('muestra 404 y permite reintentar errores', async () => {
    server.use(http.get('*/api/funnel/cycles/:id/receipt', () => new HttpResponse(null, { status: 404 })))
    const { rerender } = render(<CycleReceiptScreen id="missing" navigate={() => {}} />)
    expect(await screen.findByText('No existe el ciclo missing.')).toBeTruthy()
    server.use(http.get('*/api/funnel/cycles/:id/receipt', () => HttpResponse.error()))
    rerender(<CycleReceiptScreen id="2026-10" navigate={() => {}} />)
    expect(await screen.findByText('No pudimos cargar el recibo.')).toBeTruthy()
    server.use(http.get('*/api/funnel/cycles/:id/receipt', () => HttpResponse.json(cycleReceipts[0])))
    fireEvent.click(screen.getByRole('button', { name: 'Reintentar' }))
    expect(await screen.findByText(/0123456789ab/)).toBeTruthy()
  })
})
