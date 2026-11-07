import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const lifecycle = vi.hoisted(() => ({ points: [] as unknown[], options: null as unknown, remove: vi.fn(), setData: vi.fn((points: unknown[]) => { lifecycle.points = points }) }))
vi.mock('lightweight-charts', () => ({ createChart: () => ({ addCandlestickSeries: (options: unknown) => { lifecycle.options = options; return { setData: lifecycle.setData } }, remove: lifecycle.remove }) }))
import { Candles } from './Candles'
import { ThemeProvider } from '../ThemeContext'

describe('Candles', () => {
  afterEach(() => { cleanup(); lifecycle.points = []; lifecycle.options = null; lifecycle.remove.mockClear(); lifecycle.setData.mockClear() })
  it('passes candle OHLC values and hollow/filled theme colors, then removes the chart', async () => { const candles = [{ date: '2026-01-02', open: 1, high: 4, low: 0.5, close: 3 }]; const view = render(<ThemeProvider><Candles candles={candles} range="3m" /></ThemeProvider>); await waitFor(() => expect(lifecycle.setData).toHaveBeenCalled()); expect(lifecycle.points).toEqual([{ time: '2026-01-02', open: 1, high: 4, low: 0.5, close: 3 }]); expect(lifecycle.options).toMatchObject({ upColor: 'rgba(0,0,0,0)', downColor: '#B3261E' }); view.unmount(); expect(lifecycle.remove).toHaveBeenCalledOnce() })
})
