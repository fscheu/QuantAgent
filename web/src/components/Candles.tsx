import { useEffect, useRef } from 'react'
import type { DailyCandle } from '../api/types'
import { useTheme } from '../ThemeContext'
import { candleTheme } from '../styles/tokens'

export function Candles({ candles, range }: { candles: DailyCandle[]; range: string }) {
  const host = useRef<HTMLDivElement>(null)
  const { theme } = useTheme()
  useEffect(() => {
    let disposed = false, chart: { remove: () => void; addCandlestickSeries: (options: ReturnType<typeof candleTheme>['candle']) => { setData: (data: { time: string; open: number; high: number; low: number; close: number }[]) => void } } | undefined
    import('lightweight-charts').then(({ createChart }) => {
      if (disposed || !host.current) return
      chart = createChart(host.current, { width: host.current.clientWidth, height: 260 })
      chart.addCandlestickSeries(candleTheme(theme).candle).setData(candles.map(({ date: time, open, high, low, close }) => ({ time, open, high, low, close })))
    })
    return () => { disposed = true; chart?.remove() }
  }, [candles, range, theme])
  return <div ref={host} aria-label={`Velas, rango ${range}`} />
}
