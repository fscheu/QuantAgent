import { useState } from 'react'
import type { EtfRow } from '../api/types'
import { useEtfCandles } from '../api/hooks'
import { marketConfig } from '../config'
import { num } from '../lib/format'
import { Delta, ErrorState, Skeleton } from './index'
import { Candles } from './Candles'
import './EtfDetail.css'
import { pct } from '../lib/format'

const labels = { '3m': '3M', '6m': '6M', '1y': '1A' } as const
export function EtfDetail({ symbol, row, navigate }: { symbol: string; row?: EtfRow; navigate: (path: string) => void }) {
  const [range, setRange] = useState<typeof marketConfig.candleRanges[number]>('6m')
  const [scale, setScale] = useState<'linear' | 'log'>('linear')
  const result = useEtfCandles(symbol, range)
  const close = result.status === 'ok' ? result.data.at(-1) : undefined
  const previous = result.status === 'ok' ? result.data.at(-2) : undefined
  const delta = close && previous ? close.close - previous.close : null
  return <aside className="qa-etf-detail" aria-label={`Detalle de ${symbol}`}>
    <header><button type="button" onClick={() => navigate('/mercado')} aria-label="Cerrar detalle">‹ Mercado</button><h1>{symbol}{row ? ` · ${row.name}` : ''}</h1></header>
    {row?.held_in.length ? <p>En cartera hoy · {row.held_in.map(account => account.account_name).join(', ')}</p> : null}
    {row && <><dl className="qa-etf-returns">{([['d1', '1d'], ['w1', '1s'], ['m1', '1m'], ['ytd', 'Año']] as const).map(([key, label]) => <div key={key}><dt>{label}</dt><dd title={row.returns[key] == null ? `${row.history_sessions} sesiones disponibles` : undefined}>{row.returns[key] == null ? '—' : pct(row.returns[key]!)}</dd></div>)}</dl>{row.close != null && <p>{symbol} cerró en {num(row.close)}{row.returns.d1 == null ? '' : `, ${pct(row.returns.d1)} en el día`}{row.returns.m1 == null ? '' : `; 1 mes ${pct(row.returns.m1)}`}</p>}</>}
    <div className="qa-etf-period" role="group" aria-label="Período">{marketConfig.candleRanges.map(value => <button type="button" key={value} aria-pressed={range === value} onClick={() => setRange(value)}>{labels[value]}</button>)}</div>
    <div className="qa-etf-scale" role="group" aria-label="Escala"><button type="button" aria-pressed={scale === 'linear'} onClick={() => setScale('linear')}>Lineal</button><button type="button" aria-pressed={scale === 'log'} onClick={() => setScale('log')}>Log</button></div>
    {result.status === 'loading' ? <Skeleton label="Cargando velas" /> : result.status === 'error' ? <ErrorState message="No pudimos cargar el historial del ETF." action={{ label: 'Reintentar', onClick: result.reload }} /> : result.data.length === 0 ? <p role="status">{symbol} no tiene historia disponible.</p> : <>
      {close && <section className="qa-etf-last"><strong>{num(close.close)}</strong><Delta value={delta} missingLabel="Sin cierre anterior" /><p>Último dato: {close.date}</p></section>}
      {result.data.length < 2 ? <p role="status">Sin datos suficientes para el gráfico.</p> : <Candles candles={result.data} range={range} scale={scale} />}
      <table><caption>Datos diarios de {symbol}; vela hueca = cierre igual o mayor a apertura, vela llena = cierre menor a apertura.</caption><thead><tr><th>Fecha</th><th>Apertura</th><th>Máximo</th><th>Mínimo</th><th>Cierre</th><th>Vela</th></tr></thead><tbody>{result.data.map(candle => <tr key={candle.date}><th scope="row">{candle.date}</th><td>{num(candle.open)}</td><td>{num(candle.high)}</td><td>{num(candle.low)}</td><td>{num(candle.close)}</td><td>{candle.close >= candle.open ? 'Hueca · alcista' : 'Llena · bajista'}</td></tr>)}</tbody></table>
    </>}
  </aside>
}

