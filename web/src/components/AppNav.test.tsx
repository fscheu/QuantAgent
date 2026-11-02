import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AppNav } from './AppNav'

describe('AppNav', () => {
  afterEach(cleanup)
  it('shows all five destinations and keeps future sections inactive', () => {
    render(<AppNav path="/" navigate={vi.fn()} />)
    expect(screen.getByRole('navigation', { name: 'Secciones' })).toBeTruthy()
    for (const label of ['Inicio', 'Cuenta', 'Backtests', 'Embudo', 'Mercado']) expect(screen.getByText(label)).toBeTruthy()
    expect(screen.getAllByText('pronto')).toHaveLength(2)
    expect(screen.queryByRole('link', { name: /Embudo/ })).toBeNull()
    expect(screen.queryByRole('link', { name: /Mercado/ })).toBeNull()
  })

  it.each([
    ['/', 'Inicio'], ['/cuenta/demo', 'Cuenta'], ['/cuenta/demo/rendimiento', 'Cuenta'],
    ['/backtests', 'Backtests'], ['/rendimiento/backtest/demo', 'Backtests'],
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
})
