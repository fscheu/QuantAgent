import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { Embudo } from './Embudo'
import { handlers } from '../mocks/handlers'
import { hypotheses } from '../mocks/data'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { server.resetHandlers(); cleanup(); window.history.replaceState(null, '', '/') })
afterAll(() => server.close())

describe('Embudo', () => {
  it('carga resumen, motivos e hipótesis y admite idea nula', async () => {
    render(<Embudo navigate={() => {}} />)
    expect(await screen.findByText(/hipótesis · 3 promovidas/)).toBeTruthy()
    expect(await screen.findByText('Sharpe bajo el umbral')).toBeTruthy()
    expect(await screen.findByText(hypotheses[0].name)).toBeTruthy()
  })
  it('mantiene visibles las demás secciones si falla el embudo y muestra listas vacías', async () => {
    server.use(http.get('*/api/funnel', () => HttpResponse.error()), http.get('*/api/hypotheses', () => HttpResponse.json([])))
    render(<Embudo navigate={() => {}} />)
    expect(await screen.findByText('No pudimos cargar resumen del embudo.')).toBeTruthy()
    expect(await screen.findByText('No hay hipótesis para estos filtros.')).toBeTruthy()
    expect(screen.getAllByText(/ciclos seguidos sin promovida/).length).toBeGreaterThan(0)
  })
  it('deshabilita el período al elegir ciclo y sincroniza la URL', async () => {
    render(<Embudo navigate={() => {}} />)
    await screen.findByRole('option', { name: /2026-11/ })
    fireEvent.change(screen.getByRole('combobox', { name: 'Ciclo' }), { target: { value: '2026-11' } })
    await waitFor(() => expect(window.location.search).toContain('ciclo=2026-11'))
    expect(screen.getByRole('group', { name: /Período/ }).getAttribute('disabled')).not.toBeNull()
  })
  it('navega desde las tarjetas de hipótesis y ciclo a sus fichas', async () => {
    const navigate = vi.fn()
    render(<Embudo navigate={navigate} />)
    fireEvent.click(await screen.findByText(hypotheses[0].name))
    expect(navigate).toHaveBeenCalledWith(`/hipotesis/${hypotheses[0].id}`)
    fireEvent.click(screen.getAllByRole('link', { name: /2026-11/ })[0])
    expect(navigate).toHaveBeenCalledWith('/embudo/ciclo/2026-11')
  })
})
