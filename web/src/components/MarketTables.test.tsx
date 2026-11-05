import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { EtfRow } from '../api/types'
import { ThemeProvider } from '../ThemeContext'
import { CompareChart, EtfTable } from './MarketTables'

const row = (symbol: string, asset_class: EtfRow['asset_class'] = 'bonds'): EtfRow => ({ symbol, name: symbol, asset_class, close: 100, returns: { d1: null, w1: 1, m1: 2, m3: 3, ytd: 4 }, history_sessions: 3, in_universe_since: '2026-01-01', spark_30d: [90, 100], held_in: [] })
const themed = (ui: React.ReactNode) => render(<ThemeProvider>{ui}</ThemeProvider>)
afterEach(cleanup)

describe('Market tables', () => {
  it('groups assets, shows missing returns, and prevents selecting a sixth ETF', () => {
    const onSelectionChange = vi.fn()
    themed(<EtfTable etfs={[row('SPY', 'us_equity'), ...['A', 'B', 'C', 'D', 'E', 'F'].map(symbol => row(symbol))]} selected={['A', 'B', 'C', 'D', 'E']} onSelectionChange={onSelectionChange} navigate={vi.fn()} />)
    expect(screen.getByText('Renta variable de EE. UU.')).toBeTruthy()
    expect(screen.getByText('Bonos')).toBeTruthy()
    expect(screen.getAllByText('—', { selector: '[role="cell"]' }).length).toBeGreaterThan(0)
    expect(screen.getByRole('checkbox', { name: 'Comparar F' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('status').textContent).toMatch(/Máximo de 5/)
  })
  it('omits empty compare series and changes the selected range', () => {
    themed(<CompareChart series={[{ symbol: 'SPY', points: [{ date: '2026-11-01', value: 10 }, { date: '2026-11-03', value: 11 }] }, { symbol: 'EMPTY', points: [] }]} />)
    expect(screen.getByRole('note').textContent).toMatch(/EMPTY/)
    fireEvent.change(screen.getByLabelText('Rango de comparación'), { target: { value: '3m' } })
    expect((screen.getByLabelText('Rango de comparación') as HTMLSelectElement).value).toBe('3m')
    expect(screen.getByRole('columnheader', { name: 'SPY' })).toBeTruthy()
  })
})
