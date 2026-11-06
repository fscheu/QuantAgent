import { useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { AssetClass, EtfRow } from '../api/types'
import { marketConfig } from '../config'
import { num } from '../lib/format'
import { useTheme } from '../ThemeContext'
import { chartSeries, palette, seriesDash } from '../styles/tokens'
import { Sparkline } from './Charts'
import './MarketTables.css'

const classes: { key: AssetClass; label: string }[] = [
  { key: 'us_equity', label: 'Renta variable de EE. UU.' }, { key: 'sector', label: 'Sectores' }, { key: 'international', label: 'Internacional' },
  { key: 'bonds', label: 'Bonos' }, { key: 'commodities', label: 'Materias primas' }, { key: 'currency', label: 'Moneda' }, { key: 'real_estate', label: 'Inmobiliario' },
]
const periods = [['d1', '1d'], ['w1', '1s'], ['m1', '1m'], ['m3', '3m'], ['ytd', 'YTD']] as const

export function EtfTable({ etfs, selected, onSelectionChange, navigate }: { etfs: EtfRow[]; selected: string[]; onSelectionChange: (symbols: string[]) => void; navigate: (path: string) => void }) {
  const full = selected.length >= marketConfig.maxCompare
  const toggle = (symbol: string) => onSelectionChange(selected.includes(symbol) ? selected.filter(item => item !== symbol) : [...selected, symbol])
  return <div className="qa-etf-table" role="table" aria-label="ETFs por clase de activo">
    <div role="row" className="qa-etf-row qa-etf-head"><span role="columnheader">Comparar · ETF</span><span role="columnheader">Cierre</span>{periods.map(([, label]) => <span role="columnheader" key={label}>{label}</span>)}<span role="columnheader">30 días</span></div>
    {classes.map(group => { const rows = etfs.filter(row => row.asset_class === group.key); return rows.length ? <details key={group.key} open><summary>{group.label} <span>({rows.length})</span></summary>{rows.map(row => <div role="row" className="qa-etf-row" key={row.symbol} onClick={() => navigate(`/mercado/etf/${encodeURIComponent(row.symbol)}`)} onKeyDown={event => { if (event.key === 'Enter') navigate(`/mercado/etf/${encodeURIComponent(row.symbol)}`) }} tabIndex={0}>
      <span role="cell" className="qa-etf-name"><input type="checkbox" aria-label={`Comparar ${row.symbol}`} checked={selected.includes(row.symbol)} disabled={!selected.includes(row.symbol) && full} onClick={event => event.stopPropagation()} onChange={() => toggle(row.symbol)} /><b>{row.symbol}</b> · {row.name}{row.held_in.length > 0 && <small>En cartera · {row.held_in.map(item => item.account_name).join(', ')}</small>}</span>
      <span role="cell">{row.close == null ? '—' : num(row.close)}</span>{periods.map(([key]) => <span role="cell" key={key} title={row.returns[key] == null ? `${row.history_sessions} sesiones disponibles` : undefined}>{row.returns[key] == null ? '—' : `${row.returns[key]! > 0 ? '+' : ''}${num(row.returns[key]!)}%`}</span>)}
      <span role="cell">{row.spark_30d ? <Sparkline values={row.spark_30d} label={`${row.symbol}, últimos 30 días`} /> : '—'}</span>
    </div>)}</details> : null })}
    {full && <p role="status" className="qa-etf-limit">Máximo de {marketConfig.maxCompare} ETFs para comparar. Quitá uno para elegir otro.</p>}
  </div>
}

export type CompareSeries = { symbol: string; points: { date: string; value: number }[] }
export function CompareChart({ series }: { series: CompareSeries[] }) {
  const [range, setRange] = useState<'1m' | '3m' | '1y'>(marketConfig.defaultCompareRange)
  const { theme } = useTheme()
  const colors = chartSeries[theme]
  const empty = series.filter(item => !item.points.length)
  const available = series.filter(item => item.points.length)
  const rows = useMemo(() => {
    const cutoff = new Date(); cutoff.setMonth(cutoff.getMonth() - (range === '1m' ? 1 : range === '3m' ? 3 : 12))
    const map = new Map<string, Record<string, number | string>>()
    available.forEach(item => { const points = item.points.filter(point => new Date(`${point.date}T00:00:00`) >= cutoff); const base = points[0]?.value; points.forEach(point => { const row = map.get(point.date) ?? { date: point.date }; if (base) row[item.symbol] = point.value / base * 100; map.set(point.date, row) }) })
    return [...map.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)))
  }, [available, range])
  return <section className="qa-compare-chart"><label>Rango <select aria-label="Rango de comparación" value={range} onChange={event => setRange(event.target.value as typeof range)}><option value="1m">1m</option><option value="3m">3m</option><option value="1y">1a</option></select></label>
    {empty.map(item => <p role="note" key={item.symbol}>{item.symbol}: sin datos en este período.</p>)}
    {!available.length ? <p>Sin series con datos para comparar.</p> : <><div className="qa-compare-plot"><ResponsiveContainer width="100%" height="100%"><LineChart data={rows}><CartesianGrid vertical={false} stroke={palette[theme].grid} /><XAxis dataKey="date" /><YAxis domain={['auto', 'auto']} /><Tooltip /><Line dataKey="base" stroke={palette[theme].axis} strokeDasharray="4 4" dot={false} isAnimationActive={false} />{available.map((item, index) => <Line key={item.symbol} dataKey={item.symbol} name={item.symbol} stroke={colors[index % colors.length]} strokeDasharray={seriesDash[index % seriesDash.length]} dot={false} connectNulls isAnimationActive={false} />)}</LineChart></ResponsiveContainer></div>
      <ul className="qa-compare-legend">{available.map((item, index) => <li key={item.symbol}><span style={{ borderColor: colors[index % colors.length], borderTopStyle: seriesDash[index % seriesDash.length] ? 'dashed' : 'solid' }} />{item.symbol}</li>)}</ul>
      <table className="qa-compare-data"><caption>Índice base 100 por fecha</caption><thead><tr><th>Fecha</th>{available.map(item => <th key={item.symbol}>{item.symbol}</th>)}</tr></thead><tbody>{rows.map(row => <tr key={String(row.date)}><th>{row.date}</th>{available.map(item => <td key={item.symbol}>{typeof row[item.symbol] === 'number' ? num(row[item.symbol] as number) : '—'}</td>)}</tr>)}</tbody></table></>}
  </section>
}
