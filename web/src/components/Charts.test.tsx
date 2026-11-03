import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ThemeProvider } from '../ThemeContext'
import type { ReactNode } from 'react'
import { EquityChart, DivergingBar, Sparkline } from './Charts'
import { equity, sectorExposure } from '../mocks/data'

const themed = (ui: ReactNode) => render(<ThemeProvider>{ui}</ThemeProvider>)

describe('charts', () => {
  afterEach(cleanup)
  it('keeps every chart alternative table inside an overflow-clipped visually hidden wrapper', () => {
    themed(<><EquityChart data={equity} /><DivergingBar data={sectorExposure} /></>)
    for (const table of screen.getAllByRole('table')) {
      expect(table.classList.contains('qa-visually-hidden')).toBe(false)
      expect(table.parentElement?.classList.contains('qa-visually-hidden')).toBe(true)
    }
  })

  it('renders equity values in the accessible text table', () => {
    themed(<EquityChart data={equity} />)
    expect(screen.getAllByRole('img').some(image => /cuenta subió/.test(image.getAttribute('aria-label') ?? ''))).toBe(true)
    expect(screen.getByRole('table', { name: 'Valores de la curva de valor' })).toBeTruthy()
    expect(screen.getByText('USD 13.000,00')).toBeTruthy()
    expect(screen.getByText('USD 693,00')).toBeTruthy()
    expect(screen.getByRole('img', { name: /cuenta subió \+2,77 %; SPY, \+1,76 %; sombra, \+2,22 %/ })).toBeTruthy()
  })

  it('renders sector values in the accessible text table', () => {
    themed(<DivergingBar data={sectorExposure} />)
    expect(screen.getAllByRole('img').some(image => /Tecnología/.test(image.getAttribute('aria-label') ?? ''))).toBe(true)
    expect(screen.getByRole('table', { name: /Exposición por sector/ })).toBeTruthy()
    expect(screen.getByText('USD 4.200,00')).toBeTruthy()
    expect(screen.getAllByText('+50,00 %')).toHaveLength(2)
  })

  it('describes sparkline mock values accessibly', () => {
    themed(<Sparkline values={[12, 13, 15]} label="Variación" tone="positive" />)
    expect(screen.getByRole('img', { name: 'Variación: 12,00, 13,00, 15,00' })).toBeTruthy()
  })

  it('shows empty states for empty data', () => {
    themed(<><EquityChart data={[]} /><DivergingBar data={[]} /></>)
    expect(screen.getByText('Sin datos de valor')).toBeTruthy()
    expect(screen.getByText('Sin exposición por sector')).toBeTruthy()
  })
})
