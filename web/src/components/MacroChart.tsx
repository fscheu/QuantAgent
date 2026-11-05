import { useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { MacroSeries, RecessionPeriod } from '../api/types'
import { marketConfig } from '../config'
import { useTheme } from '../ThemeContext'
import { num } from '../lib/format'
import { palette } from '../styles/tokens'
import { SegmentedControl } from './index'
import './MacroChart.css'

type Range = typeof marketConfig.macroRanges[number]
const options = [{ value: '1y', label: '1 año' }, { value: '5y', label: '5 años' }, { value: 'all', label: 'Todo' }]
const cutoff = (range: Range) => range === 'all' ? '' : new Date(Date.now() - (range === '1y' ? 365 : 1825) * 86400000).toISOString().slice(0, 10)

export function MacroLineChart({ series, recessions, range, onRangeChange }: { series: MacroSeries; recessions: RecessionPeriod[]; range?: Range; onRangeChange?: (range: Range) => void }) {
  const { theme } = useTheme()
  const [selected, setSelected] = useState<Range>(() => range ?? (typeof window !== 'undefined' && window.matchMedia?.('(max-width: 639px)').matches ? marketConfig.defaultMacroRange.phone : marketConfig.defaultMacroRange.desktop))
  const activeRange = range ?? selected
  const points = useMemo(() => { const start = cutoff(activeRange); return series.points.filter(point => !start || point.date >= start) }, [series.points, activeRange])
  const visibleRecessions = recessions.map(period => ({ start: period.start < (points[0]?.date ?? '') ? points[0]?.date ?? period.start : period.start, end: period.end > (points.at(-1)?.date ?? '') ? points.at(-1)?.date ?? period.end : period.end })).filter(period => points.length && period.end >= points[0].date && period.start <= points.at(-1)!.date)
  const setRange = (value: string) => { const next = value as Range; setSelected(next); onRangeChange?.(next) }
  return <section className="qa-macro-line-chart" aria-label={series.title}>
    <SegmentedControl label="Rango macro" options={options} value={activeRange} onChange={setRange} />
    {series.stale && <p role="status" className="qa-macro-chart-state">Dato desactualizado.</p>}
    {!points.length ? <p role="status" className="qa-macro-chart-state">Sin datos para este rango.</p> : <>
      <div className="qa-macro-chart-plot" role="img" aria-label={`${series.title}, ${points.length} observaciones`}><ResponsiveContainer width="100%" height="100%"><LineChart data={points}>
        <CartesianGrid vertical={false} stroke={palette[theme].grid} />
        <XAxis dataKey="date" tick={{ fill: palette[theme].axis, fontSize: 11 }} />
        <YAxis tick={{ fill: palette[theme].axis, fontSize: 11 }} />
        <Tooltip formatter={(value) => [`${num(Number(value))} ${series.unit}`, series.title]} />
        {visibleRecessions.map(period => <ReferenceArea key={`${period.start}-${period.end}`} x1={period.start} x2={period.end} fill={palette[theme].neg} fillOpacity={0.12} />)}
        <Line dataKey="value" name={series.title} stroke={palette[theme].accent} dot={false} isAnimationActive={false} />
      </LineChart></ResponsiveContainer></div>
      {!visibleRecessions.length && <p className="qa-macro-chart-state">No hubo recesiones en este rango.</p>}
      <span className="qa-visually-hidden" data-recessions={visibleRecessions.map(period => `${period.start}:${period.end}`).join(',')} />
      <table><caption>{series.title}: valores observados</caption><thead><tr><th>Fecha</th><th>Valor ({series.unit})</th></tr></thead><tbody>{points.map(point => <tr key={point.date}><th>{point.date}</th><td>{num(point.value)}</td></tr>)}</tbody></table>
    </>}
  </section>
}
