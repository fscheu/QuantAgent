import { useState } from 'react'
import type { ReactNode } from 'react'
import { useMarket, useMarketSeries } from '../api/hooks'
import { marketConfig } from '../config'
import { ErrorState, Skeleton } from '../components'
import { MacroLineChart } from '../components/MacroChart'
import { EtfDetail } from '../components/EtfDetail'
import { MacroIndicatorCard, MarketAsOfLine, MotorStateCard, RegimeCard, SectorHeatmap, SectorTable } from '../components/Market'
import { CompareChart, EtfTable } from '../components/MarketTables'
import './Mercado.css'

export function Mercado({ symbol, navigate }: { navigate: (path: string) => void; symbol?: string }) {
  const market = useMarket(), [selected, setSelected] = useState<string[]>([...marketConfig.defaultCompare]), [sectorView, setSectorView] = useState<'table' | 'heat'>('table')
  const series = useMarketSeries(selected, marketConfig.defaultCompareRange)
  if (symbol) return <EtfDetail symbol={decodeURIComponent(symbol)} row={market.status === 'ok' ? market.data.etfs.find(row => row.symbol === decodeURIComponent(symbol)) : undefined} navigate={navigate} />
  if (market.status === 'loading') return <main className="qa-mercado"><h1>Mercado</h1><Skeleton label="Mercado" /></main>
  if (market.status === 'error') return <main className="qa-mercado"><h1>Mercado</h1><ErrorState message="No pudimos cargar el mercado." action={{ label: 'Reintentar', onClick: market.reload }} /></main>
  const data = market.data
  const failed = (key: string, value: unknown) => data.errors?.[key] || value == null
  const section = (title: string, key: string, value: unknown, content: ReactNode) => <section className="qa-mercado-section" key={key}><h2>{title}</h2>{failed(key, value) ? <ErrorState message={data.errors?.[key] ?? `No hay datos para ${title.toLowerCase()}.`} action={{ label: 'Reintentar', onClick: market.reload }} /> : content}</section>
  const etfs = data.etfs ?? []
  return <main className="qa-mercado">
    <header><h1>Mercado</h1>{section('Actualización', 'as_of', data.as_of, <MarketAsOfLine asOf={data.as_of} />)}</header>
    <div className="qa-mercado-top">{section('Régimen', 'regime', data.regime, <RegimeCard regime={data.regime} stale={data.as_of.stale} />)}{section('Motor', 'motor', true, <MotorStateCard motor={data.motor} sectors={data.sectors ?? []} />)}</div>
    {section('Sectores', 'sectors', data.sectors, <><div className="qa-mercado-toggle"><button aria-pressed={sectorView === 'table'} onClick={() => setSectorView('table')}>Tabla</button><button aria-pressed={sectorView === 'heat'} onClick={() => setSectorView('heat')}>Mapa de calor</button></div>{sectorView === 'table' ? <SectorTable sectors={data.sectors} /> : <SectorHeatmap sectors={data.sectors} />}</>)}
    {section('Macro', 'macro', data.macro, <div className="qa-mercado-macro">{(data.macro ?? []).map(item => <div key={item.code}>{item.status === 'active' ? <><MacroIndicatorCard series={item} /><MacroLineChart series={item} recessions={data.recessions} /></> : <MacroIndicatorCard series={item} />}</div>)}</div>)}
    {section('ETFs', 'etfs', data.etfs, <div className="qa-mercado-etfs"><EtfTable etfs={etfs} selected={selected} onSelectionChange={setSelected} navigate={navigate} />{series.status === 'loading' ? <Skeleton label="Comparación de ETFs" /> : series.status === 'error' ? <ErrorState message="No pudimos cargar la comparación." action={{ label: 'Reintentar', onClick: series.reload }} /> : <CompareChart series={series.data} />}</div>)}
  </main>
}
