import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { handlers } from '../mocks/handlers'
import { Cuenta } from './Cuenta'
import { ThemeProvider } from '../ThemeContext'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { cleanup(); server.resetHandlers() })
afterAll(() => server.close())
const navigate = () => {}
const cuenta = (accountId: string) => <ThemeProvider><Cuenta accountId={accountId} navigate={navigate} /></ThemeProvider>

describe('Cuenta', () => {
  it('muestra resumen, posición corta y orden rechazada con motivo', async () => {
    render(cuenta('paper-m4'))
    expect(await screen.findAllByText('Poder de compra insuficiente')).toBeTruthy()
    expect(screen.getAllByText('Corta')[0]).toBeTruthy()
    expect(screen.getByText('13.000,00')).toBeTruthy()
  })

  it('muestra las tres secciones nuevas con datos mock', async () => {
    render(cuenta('paper-m4'))
    expect(await screen.findByText(/En 5 días la cuenta/)).toBeTruthy()
    expect(await screen.findAllByText('Tecnología')).not.toHaveLength(0)
    expect(await screen.findByText('SPY, MSFT')).toBeTruthy()
    expect(screen.queryByLabelText('Curva de valor, próximamente')).toBeNull()
  })

  it('permite reintentar una sección que falla por red', async () => {
    let failed = false
    server.use(http.get('*/api/accounts/paper-m4/dividends', () => {
      if (!failed) { failed = true; return HttpResponse.error() }
      return HttpResponse.json([])
    }))
    render(cuenta('paper-m4'))
    const section = await screen.findByText('No pudimos cargar los dividendos.')
    expect(section).toBeTruthy()
    const retry = section.parentElement?.querySelector('button')
    if (!retry) throw new Error('Falta acción de reintento')
    fireEvent.click(retry)
    expect(await screen.findByText('Sin dividendos')).toBeTruthy()
  })

  it('para no_data reemplaza la pantalla y no muestra cifras en cero', async () => {
    render(cuenta('live-main'))
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
    render(cuenta('paper-m4'))
    const retry = await screen.findByRole('button', { name: 'Reintentar' })
    fireEvent.click(retry)
    expect(await screen.findByText('Sin posiciones')).toBeTruthy()
  })

  it('el filtro Rechazadas deja únicamente órdenes rechazadas', async () => {
    render(cuenta('paper-m4'))
    await screen.findAllByText('Poder de compra insuficiente')
    fireEvent.click(screen.getByRole('button', { name: /Rechazadas/ }))
    await waitFor(() => expect(screen.queryByText('NVDA')).toBeNull())
    expect(screen.getAllByText('AAPL').length).toBeGreaterThan(0)
  })

  it('muestra vacío cuando no hay órdenes para la fecha', async () => {
    server.use(http.get('*/api/accounts/paper-m4/orders', () => HttpResponse.json([])))
    render(<ThemeProvider><Cuenta accountId="paper-m4" navigate={navigate} /></ThemeProvider>)
    expect(await screen.findByText('Sin órdenes ese día')).toBeTruthy()
  })
})
