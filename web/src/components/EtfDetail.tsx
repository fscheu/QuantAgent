import { useState } from 'react'
import type { EtfRow } from '../api/types'
import { useEtfCandles } from '../api/hooks'
import { marketConfig } from '../config'
import { num } from '../lib/format'
import { Delta, ErrorState, Skeleton } from './index'
import { Candles } from './Candles'
import './EtfDetail.css'

const labels = { '3m': '3M', '6m': '6M', '1y': '1A' } as const
export function EtfDetail({ symbol, row, navigate }: { symbol: string; row?: EtfRow; navigate: (path: string) => void }) {
  const [range, setRange] = useState<typeof marketConfig.candleRanges[number]>('6m')
  const result = useEtfCandles(symbol, range)
  const close = result.status === 'ok' ? result.data.at(-1) : undefined
  const previous = result.status === 'ok' ? result.data.at(-2) : undefined
  const delta = close && previous ? close.close - previous.close : null
  return <aside className="qa-etf-detail" aria-label={`Detalle de ${symbol}`}>
    <header><button type="button" onClick={() => navigate('/mercado')} aria-label="Cerrar detalle">‹ Mercado</button><h1>{symbol}{row ? ` · ${row.name}` : ''}</h1></header>
    {row?.held_in.length ? <p>En cartera hoy · {row.held_in.map(account => account.account_name).join(', ')}</p> : null}
    <div className="qa-etf-period" role="group" aria-label="Período">{marketConfig.candleRanges.map(value => <button type="button" key={value} aria-pressed={range === value} onClick={() => setRange(value)}>{labels[value]}</button>)}</div>
    {result.status === 'loading' ? <Skeleton label="Cargando velas" /> : result.status === 'error' ? <ErrorState message="No pudimos cargar el historial del ETF." action={{ label: 'Reintentar', onClick: result.reload }} /> : result.data.length === 0 ? <p role="status">{symbol} no tiene historia disponible.</p> : <>
      {close && <section className="qa-etf-last"><strong>{num(close.close)}</strong><Delta value={delta} missingLabel="Sin cierre anterior" /><p>Último dato: {close.date}</p></section>}
      {result.data.length < 2 ? <p role="status">Sin datos suficientes para el gráfico.</p> : <Candles candles={result.data} range={range} />}
      <table><caption>Datos diarios de {symbol}; vela hueca = cierre igual o mayor a apertura, vela llena = cierre menor a apertura.</caption><thead><tr><th>Fecha</th><th>Apertura</th><th>Máximo</th><th>Mínimo</th><th>Cierre</th><th>Vela</th></tr></thead><tbody>{result.data.map(candle => <tr key={candle.date}><th scope="row">{candle.date}</th><td>{num(candle.open)}</td><td>{num(candle.high)}</td><td>{num(candle.low)}</td><td>{num(candle.close)}</td><td>{candle.close >= candle.open ? 'Hueca · alcista' : 'Llena · bajista'}</td></tr>)}</tbody></table>
    </>}
  </aside>
}

