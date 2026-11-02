import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ThemeProvider } from '../ThemeContext'
import type { ReactNode } from 'react'
import { EquityChart, DivergingBar, Sparkline } from './Charts'
import { equity, sectorExposure } from '../mocks/data'

const themed = (ui: ReactNode) => render(<ThemeProvider>{ui}</ThemeProvider>)

describe('charts', () => {
  it('renders equity values in the accessible text table', () => {
    themed(<EquityChart data={equity} />)
    expect(screen.getByRole('img', { name: /cuenta subió/ })).toBeTruthy()
    expect(screen.getByRole('table', { name: 'Valores de la curva de valor' })).toBeTruthy()
    expect(screen.getByText('USD 13.000,00')).toBeTruthy()
    expect(screen.getByText('USD 693,00')).toBeTruthy()
  })

  it('renders sector values in the accessible text table', () => {
    themed(<DivergingBar data={sectorExposure} />)
    expect(screen.getByRole('img', { name: /Tecnología/ })).toBeTruthy()
    expect(screen.getByRole('table', { name: /Exposición por sector/ })).toBeTruthy()
    expect(screen.getByText('+42,00 %')).toBeTruthy()
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
