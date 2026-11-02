import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { EquityPoint, SectorExposure } from '../api/types'
import { useTheme } from '../ThemeContext'
import { chartSeries, palette, seriesDash } from '../styles/tokens'
import { money, num, pct } from '../lib/format'
import { EmptyState, SegmentedControl, Skeleton } from './index'
import './Charts.css'

export type EquityRange = '1W' | '1M' | '3M' | 'YTD' | 'ALL'
const ranges: { value: EquityRange; label: string }[] = [
  { value: '1W', label: '1W' }, { value: '1M', label: '1M' }, { value: '3M', label: '3M' }, { value: 'YTD', label: 'YTD' }, { value: 'ALL', label: 'ALL' },
]
const visible = (rows: EquityPoint[], key: 'benchmark' | 'shadow') => rows.some(row => row[key] !== null)
const change = (rows: EquityPoint[], key: 'account' | 'benchmark' | 'shadow') => {
  const first = rows.find(row => row[key] !== null)?.[key]
  const last = [...rows].reverse().find(row => row[key] !== null)?.[key]
  return first && last !== null && last !== undefined ? (last / first - 1) * 100 : null
}
const signed = (value: number) => `${value > 0 ? '+' : ''}${pct(value)}`

export function EquityChart({ data, range = '3M', onRangeChange, loading = false }: { data: EquityPoint[]; range?: EquityRange; onRangeChange?: (range: EquityRange) => void; loading?: boolean }) {
  const { theme } = useTheme()
  const colors = chartSeries[theme]
  const summaryValues = { account: change(data, 'account'), benchmark: change(data, 'benchmark'), shadow: change(data, 'shadow') }
  const summary = data.length ? `En ${data.length} días la cuenta ${summaryValues.account === null ? 'no tiene variación calculable' : `subió ${signed(summaryValues.account)}`}${summaryValues.benchmark === null ? '' : `; SPY, ${signed(summaryValues.benchmark)}`}${summaryValues.shadow === null ? '' : `; sombra, ${signed(summaryValues.shadow)}`}` : 'Curva de valor sin datos'
  if (loading) return <div className="qa-chart-loading" aria-label="Cargando curva de valor"><Skeleton label="Curva de valor" /></div>
  if (!data.length) return <EmptyState title="Sin datos de valor" message="No hay puntos de valor para este rango." />
  const series = [{ key: 'account', label: 'Cuenta' }, ...(visible(data, 'benchmark') ? [{ key: 'benchmark', label: 'SPY' }] : []), ...(visible(data, 'shadow') ? [{ key: 'shadow', label: 'Sombra' }] : [])] as const
  return <div className="qa-equity-chart">
    {onRangeChange && <SegmentedControl label="Rango de la curva" options={ranges} value={range} onChange={value => onRangeChange(value as EquityRange)} />}
    <p className="qa-chart-summary">{summary}</p>
    <div className="qa-chart-plot" role="img" aria-label={summary}>
      <ResponsiveContainer width="100%" height="100%"><ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 4 }}>
        <CartesianGrid vertical={false} stroke={palette[theme].grid} />
        <XAxis dataKey="date" tick={{ fill: palette[theme].axis, fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis yAxisId="right" orientation="right" tick={{ fill: palette[theme].axis, fontSize: 11 }} axisLine={false} tickLine={false} tickCount={4} tickFormatter={value => `${Math.round(Number(value) / 1000)}k`} />
        <Area yAxisId="right" type="monotone" dataKey="account" stroke="none" fill={colors[0]} fillOpacity={0.07} isAnimationActive={false} />
        {series.map(({ key, label }) => { const index = key === 'account' ? 0 : key === 'benchmark' ? 1 : 2; return <Line key={key} yAxisId="right" type="monotone" dataKey={key} name={label} stroke={colors[index]} strokeWidth={index === 0 ? 2.25 : 1.75} strokeDasharray={seriesDash[index]} dot={false} connectNulls isAnimationActive={false} /> })}
      </ComposedChart></ResponsiveContainer>
    </div>
    <table className="qa-visually-hidden"><caption>Valores de la curva de valor</caption><thead><tr><th>Fecha</th>{series.map(s => <th key={s.key}>{s.label}</th>)}</tr></thead><tbody>{data.map(row => <tr key={row.date}><th>{row.date}</th>{series.map(s => <td key={s.key}>{row[s.key as keyof EquityPoint] === null ? 'Sin dato' : money(row[s.key as keyof EquityPoint] as number)}</td>)}</tr>)}</tbody></table>
    <ul className="qa-chart-legend" aria-hidden="true">{series.map(s => { const index = s.key === 'account' ? 0 : s.key === 'benchmark' ? 1 : 2; return <li key={s.key}><span style={{ borderColor: colors[index], borderTopStyle: index ? 'dashed' : 'solid' }} />{s.label}</li> })}</ul>
  </div>
}

export function DivergingBar({ data, loading = false }: { data: SectorExposure[]; loading?: boolean }) {
  const { theme } = useTheme()
  const colors = chartSeries[theme]
  if (loading) return <div className="qa-chart-loading" aria-label="Cargando exposición"><Skeleton label="Exposición por sector" /></div>
  if (!data.length) return <EmptyState title="Sin exposición por sector" message="No hay posiciones para calcular la exposición." />
  const max = Math.max(1, ...data.flatMap(row => [Math.abs(row.short), Math.abs(row.long)]))
  const summary = `Exposición por sector: ${data.map(row => `${row.sector}, corta ${pct(Math.abs(row.short))}, larga ${pct(row.long)}`).join('; ')}`
  return <div className="qa-diverging" role="img" aria-label={summary}>
    {data.map(row => <div className="qa-diverging-row" key={row.sector}><span className="qa-diverging-sector">{row.sector}</span><div className="qa-diverging-bars"><span className="qa-diverging-left"><span className="qa-diverging-short" style={{ width: `${Math.abs(row.short) / max * 100}%`, background: colors[1] }} /></span><span className="qa-diverging-axis" /><span className="qa-diverging-right"><span className="qa-diverging-long" style={{ width: `${row.long / max * 100}%`, background: colors[0] }} /></span></div><span className="qa-diverging-value">{pct(row.pct)}</span></div>)}
    <table className="qa-visually-hidden"><caption>{summary}</caption><thead><tr><th>Sector</th><th>Short</th><th>Long</th><th>Exposición</th></tr></thead><tbody>{data.map(row => <tr key={row.sector}><th>{row.sector}</th><td>{pct(row.short)}</td><td>{pct(row.long)}</td><td>{pct(row.pct)}</td></tr>)}</tbody></table>
  </div>
}

export function Sparkline({ values, label, tone = 'neutral' }: { values: number[]; label: string; tone?: 'positive' | 'negative' | 'neutral' }) {
  const { theme } = useTheme()
  const colors = palette[theme]
  if (!values.length) return null
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${28 - ((value - min) / span) * 24}`).join(' ')
  const color = tone === 'positive' ? colors.pos : tone === 'negative' ? colors.neg : colors.text2
  return <svg className="qa-sparkline" viewBox="0 0 100 32" role="img" aria-label={`${label}: ${values.map(value => num(value)).join(', ')}`}><polyline points={points} fill="none" stroke={colors.text3} strokeWidth="1.5" vectorEffect="non-scaling-stroke" /><circle cx={points.split(' ').at(-1)?.split(',')[0]} cy={points.split(' ').at(-1)?.split(',')[1]} r="2.5" fill={color} /></svg>
}
