import { cleanup, render, screen } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { Mercado } from './Mercado'
import { ThemeProvider } from '../ThemeContext'
import { handlers } from '../mocks/handlers'
import { marketPage } from '../mocks/data'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { server.resetHandlers(); cleanup() })
afterAll(() => server.close())

describe('Mercado', () => {
  it('renderiza los bloques con los 29 ETFs y posiciones declaradas', async () => {
    render(<ThemeProvider><Mercado navigate={() => {}} /></ThemeProvider>)
    expect(await screen.findByText('Mapa de calor')).toBeTruthy()
    expect(screen.getByRole('table', { name: 'ETFs por clase de activo' })).toBeTruthy()
    expect(screen.getByText(/En cartera · Paper M4/)).toBeTruthy()
    expect(marketPage.etfs).toHaveLength(29)
    expect(marketPage.etfs.every(row => Array.isArray(row.held_in))).toBe(true)
  })
  it('muestra aviso cuando la fecha de mercado está atrasada', async () => {
    server.use(http.get('*/api/market', () => HttpResponse.json({ ...marketPage, as_of: { ...marketPage.as_of, stale: true, missing_sessions: ['2026-10-29'] } })))
    render(<ThemeProvider><Mercado navigate={() => {}} /></ThemeProvider>)
    expect(await screen.findByText('Datos de mercado desactualizados')).toBeTruthy()
  })
  it('conserva el resto de bloques si falla un endpoint del agregado', async () => {
    server.use(http.get('*/api/market', () => HttpResponse.json({ ...marketPage, macro: null, errors: { macro: 'FRED sin respuesta' } })))
    render(<ThemeProvider><Mercado navigate={() => {}} /></ThemeProvider>)
    expect(await screen.findByText('FRED sin respuesta')).toBeTruthy()
    expect(screen.getByText('ETFs')).toBeTruthy()
  })
})
