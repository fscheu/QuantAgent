import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const lifecycle = vi.hoisted(() => ({ points: [] as unknown[], remove: vi.fn(), setData: vi.fn((points: unknown[]) => { lifecycle.points = points }) }))
vi.mock('lightweight-charts', () => ({ createChart: () => ({ addCandlestickSeries: () => ({ setData: lifecycle.setData }), remove: lifecycle.remove }) }))
import { Candles } from './Candles'

describe('Candles', () => {
  afterEach(() => { cleanup(); lifecycle.points = []; lifecycle.remove.mockClear(); lifecycle.setData.mockClear() })
  it('passes candle OHLC values to the series and removes the chart on unmount', async () => { const candles = [{ date: '2026-01-02', open: 1, high: 4, low: 0.5, close: 3 }]; const view = render(<Candles candles={candles} range="3m" />); await waitFor(() => expect(lifecycle.setData).toHaveBeenCalled()); expect(lifecycle.points).toEqual([{ time: '2026-01-02', open: 1, high: 4, low: 0.5, close: 3 }]); view.unmount(); expect(lifecycle.remove).toHaveBeenCalledOnce() })
})
