import { useEffect, useRef } from 'react'
import type { DailyCandle } from '../api/types'

export function Candles({ candles, range }: { candles: DailyCandle[]; range: string }) {
  const host = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let disposed = false, chart: { remove: () => void; addCandlestickSeries: () => { setData: (data: { time: string; open: number; high: number; low: number; close: number }[]) => void } } | undefined
    import('lightweight-charts').then(({ createChart }) => {
      if (disposed || !host.current) return
      chart = createChart(host.current, { width: host.current.clientWidth, height: 260 })
      chart.addCandlestickSeries().setData(candles.map(({ date: time, open, high, low, close }) => ({ time, open, high, low, close })))
    })
    return () => { disposed = true; chart?.remove() }
  }, [candles, range])
  return <div ref={host} aria-label={`Velas, rango ${range}`} />
}
