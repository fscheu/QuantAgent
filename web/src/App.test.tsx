import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { ThemeProvider, useTheme } from './ThemeContext'

function Preferences() {
  const { setTheme, setDensity } = useTheme()
  return (
    <>
      <button onClick={() => setTheme('dark')}>Dark</button>
      <button onClick={() => setDensity('compact')}>Compact</button>
    </>
  )
}

describe('App', () => {
  afterEach(() => { window.history.replaceState(null, '', '/') })

  it.each([
    ['/embudo', 'Embudo'], ['/embudo/ciclo/ciclo-7', 'Embudo · ciclo-7'],
    ['/hipotesis/hip-2', 'Hipótesis · hip-2'], ['/mercado', 'Mercado'],
  ])('renders the route %s with its parameter', (path, title) => {
    window.history.replaceState(null, '', path)
    render(<ThemeProvider><App /></ThemeProvider>)
    expect(screen.getByRole('heading', { name: title })).toBeTruthy()
    if (path.startsWith('/embudo') || path === '/mercado') expect(screen.queryByText('En construcción')).toBeNull()
    else expect(screen.getByText('En construcción')).toBeTruthy()
  })

  it('renders the ETF detail route', () => {
    window.history.replaceState(null, '', '/mercado/etf/SPY')
    render(<ThemeProvider><App /></ThemeProvider>)
    expect(screen.getByRole('heading', { name: 'SPY' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Cerrar detalle' })).toBeTruthy()
  })

  it('falls back to Inicio for an unknown route', () => {
    window.history.replaceState(null, '', '/desconocida')
    render(<ThemeProvider><App /></ThemeProvider>)
    expect(screen.getByRole('heading', { name: 'Inicio' })).toBeTruthy()
  })

  it('renders the Inicio screen', () => {
    render(<ThemeProvider><App /></ThemeProvider>)
    expect(screen.getByRole('heading', { name: 'Inicio' })).toBeTruthy()
  })

  it('applies theme and density changes to the document', () => {
    render(<ThemeProvider><Preferences /></ThemeProvider>)
    fireEvent.click(screen.getByText('Dark'))
    fireEvent.click(screen.getByText('Compact'))
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.documentElement.dataset.density).toBe('compact')
  })

  it('renders when localStorage is unavailable', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    expect(() => render(<ThemeProvider><App /></ThemeProvider>)).not.toThrow()
    expect(screen.getByRole('heading', { name: 'Inicio' })).toBeTruthy()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    delete document.documentElement.dataset.theme
    delete document.documentElement.dataset.density
  })
})
