import { cleanup, render, screen, fireEvent } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ThemeProvider } from '../ThemeContext'
import type { MacroSeries } from '../api/types'
import { MacroLineChart } from './MacroChart'

const series: MacroSeries = { code: 'X', title: 'Serie X', frequency: 'monthly', status: 'active', unit: 'index', stale: false, expected_release: null, last: null, change_1m: null, points: [{ date: '2025-01-01', value: 1 }, { date: '2026-10-01', value: 2 }, { date: '2026-11-01', value: 3 }] }
const show = (recessions = [{ start: '2026-01-01', end: '2026-06-01' }], stale = false) => render(<ThemeProvider><MacroLineChart series={{ ...series, stale }} recessions={recessions} range="all" /></ThemeProvider>)

describe('MacroLineChart', () => {
  afterEach(cleanup)
  it('shades only recessions overlapping the selected date range', () => expect(show([{ start: '2025-12-01', end: '2026-02-01' }, { start: '2020-01-01', end: '2020-06-01' }]).container.querySelector('[data-recessions]')?.getAttribute('data-recessions')).toBe('2025-12-01:2026-02-01'))
  it('changes visible points with the range selector', () => { render(<ThemeProvider><MacroLineChart series={series} recessions={[]} /></ThemeProvider>); fireEvent.click(screen.getByRole('radio', { name: '1 año' })); expect(screen.getByRole('table').querySelectorAll('tbody tr')).toHaveLength(2) })
  it('explains empty recession periods and stale data', () => { show([], true); expect(screen.getByText('No hubo recesiones en este rango.')).toBeTruthy(); expect(screen.getByText('Dato desactualizado.')).toBeTruthy() })
})
