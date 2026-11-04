import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AppNav } from './AppNav'

describe('AppNav', () => {
  afterEach(cleanup)
  it('shows all five destinations with Embudo and Mercado enabled', () => {
    render(<AppNav path="/" navigate={vi.fn()} />)
    expect(screen.getByRole('navigation', { name: 'Secciones' })).toBeTruthy()
    for (const label of ['Inicio', 'Cuenta', 'Backtests', 'Embudo', 'Mercado']) expect(screen.getByText(label)).toBeTruthy()
    expect(screen.queryByText('pronto')).toBeNull()
    expect(screen.getByRole('link', { name: /Embudo/ })).toBeTruthy()
    expect(screen.getByRole('link', { name: /Mercado/ })).toBeTruthy()
  })

  it.each([
    ['/', 'Inicio'], ['/cuenta/demo', 'Cuenta'], ['/cuenta/demo/rendimiento', 'Cuenta'],
    ['/backtests', 'Backtests'], ['/rendimiento/backtest/demo', 'Backtests'],
    ['/embudo', 'Embudo'], ['/embudo/ciclo/c-1', 'Embudo'], ['/hipotesis/h-1', 'Embudo'],
    ['/mercado', 'Mercado'], ['/mercado/etf/SPY', 'Mercado'],
  ])('marks %s active at %s', (path, label) => {
    render(<AppNav path={path} navigate={vi.fn()} />)
    expect(screen.getByRole('link', { name: new RegExp(label) }).getAttribute('aria-current')).toBe('page')
  })

  it('uses client navigation when Backtests is activated', () => {
    const navigate = vi.fn()
    render(<AppNav path="/" navigate={navigate} />)
    fireEvent.click(screen.getByRole('link', { name: /Backtests/ }))
    expect(navigate).toHaveBeenCalledWith('/backtests')
  })

  it.each([['Embudo', '/embudo'], ['Mercado', '/mercado']])('uses client navigation for %s', (label, path) => {
    const navigate = vi.fn()
    render(<AppNav path="/" navigate={navigate} />)
    fireEvent.click(screen.getByRole('link', { name: new RegExp(label) }))
    expect(navigate).toHaveBeenCalledWith(path)
  })
})
