import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { handlers } from '../mocks/handlers'
import { Cuenta } from './Cuenta'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { cleanup(); server.resetHandlers() })
afterAll(() => server.close())
const navigate = () => {}

describe('Cuenta', () => {
  it('muestra resumen, posición corta y orden rechazada con motivo', async () => {
    render(<Cuenta accountId="paper-m4" navigate={navigate} />)
    expect(await screen.findAllByText('Poder de compra insuficiente')).toBeTruthy()
    expect(screen.getAllByText('Corta')[0]).toBeTruthy()
    expect(screen.getByText('13.000,00')).toBeTruthy()
  })

  it('para no_data reemplaza la pantalla y no muestra cifras en cero', async () => {
    render(<Cuenta accountId="live-main" navigate={navigate} />)
    expect(await screen.findByText(/todavía no tiene datos/)).toBeTruthy()
    expect(screen.queryByText('USD 0,00')).toBeNull()
    expect(screen.queryByText('Posiciones')).toBeNull()
  })

  it('reintenta solo posiciones después de un error de red', async () => {
    let failed = false
    server.use(http.get('*/api/accounts/paper-m4/positions', () => {
      if (!failed) { failed = true; return HttpResponse.error() }
      return HttpResponse.json([])
    }))
    render(<Cuenta accountId="paper-m4" navigate={navigate} />)
    const retry = await screen.findByRole('button', { name: 'Reintentar' })
    fireEvent.click(retry)
    expect(await screen.findByText('Sin posiciones')).toBeTruthy()
  })

  it('el filtro Rechazadas deja únicamente órdenes rechazadas', async () => {
    render(<Cuenta accountId="paper-m4" navigate={navigate} />)
    await screen.findAllByText('Poder de compra insuficiente')
    fireEvent.click(screen.getByRole('button', { name: /Rechazadas/ }))
    await waitFor(() => expect(screen.queryByText('NVDA')).toBeNull())
    expect(screen.getAllByText('AAPL').length).toBeGreaterThan(0)
  })

  it('muestra vacío cuando no hay órdenes para la fecha', async () => {
    server.use(http.get('*/api/accounts/paper-m4/orders', () => HttpResponse.json([])))
    render(<Cuenta accountId="paper-m4" navigate={navigate} />)
    expect(await screen.findByText('Sin órdenes ese día')).toBeTruthy()
  })
})
