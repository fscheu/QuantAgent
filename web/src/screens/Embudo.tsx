import { useEffect, useState } from 'react'
import { useCycles, useCurrentCycle, useFunnel, useHistory, useHypotheses, useReviewRule } from '../api/hooks'
import type { CycleSummary } from '../api/types'
import { funnelConfig } from '../config'
import { CycleFilter, CycleMilestones, CycleSummaryCard, DropReasons, FunnelChart, HistoryPanel, HypothesisCard, ReviewCounter } from '../components/Funnel'
import { ErrorState, Skeleton } from '../components'
import { CycleReceiptScreen } from './CycleReceipt'

type Period = '30d' | '90d' | 'all'
const params = () => new URLSearchParams(window.location.search)

export function Embudo({ navigate, ciclo: routeCycle }: { navigate: (path: string) => void; ciclo?: string }) {
  if (routeCycle) return <CycleReceiptScreen id={routeCycle} navigate={navigate} />
  const initial = params(), [cycle, setCycle] = useState(routeCycle ?? initial.get('ciclo') ?? funnelConfig.defaultCycle), [period, setPeriod] = useState<Period>((initial.get('periodo') as Period) || funnelConfig.defaultPeriod)
  useEffect(() => { const query = new URLSearchParams(); if (cycle !== funnelConfig.defaultCycle) query.set('ciclo', cycle); if (period !== funnelConfig.defaultPeriod) query.set('periodo', period); const search = query.toString(); window.history.replaceState(null, '', `${window.location.pathname}${search ? `?${search}` : ''}`) }, [cycle, period])
  const funnel = useFunnel(cycle, period), cycles = useCycles(), current = useCurrentCycle(), review = useReviewRule(), history = useHistory(), hypotheses = useHypotheses({ cycle, period })
  const chooseCycle = (value: string) => setCycle(value)
  const section = <T,>(state: { status: string; data?: T; reload: () => void }, label: string, render: (data: T) => React.ReactNode) => state.status === 'loading' ? <Skeleton label={label} /> : state.status === 'error' ? <ErrorState message={`No pudimos cargar ${label.toLowerCase()}.`} action={{ label: 'Reintentar', onClick: state.reload }} /> : render(state.data as T)
  const visibleCycle: CycleSummary | undefined = cycle !== 'all' && cycles.status === 'ok' ? cycles.data.find(item => item.cycle === cycle) : undefined
  return <main className="qa-embudo"><header><h1>Embudo{routeCycle ? ` · ${routeCycle}` : ''}</h1><CycleFilter value={cycle} cycles={cycles.status === 'ok' ? cycles.data : []} period={period} onChange={chooseCycle} onPeriodChange={setPeriod} /></header>
    <section aria-label="Regla de revisión">{review.status === 'error' ? <ReviewCounter rule={null} error /> : <ReviewCounter rule={review.status === 'ok' ? review.data : null} loading={review.status === 'loading'} />}</section>
    {cycle === 'none' ? <section aria-label="Historial importado">{section(history, 'Historial importado', data => <HistoryPanel history={data} />)}</section> : <>
      {cycle === 'all' && <section aria-label="Ciclo actual">{current.status === 'error' ? <ErrorState message="No pudimos cargar el ciclo actual." action={{ label: 'Reintentar', onClick: current.reload }} /> : current.status === 'loading' ? <Skeleton label="Ciclo actual" /> : <a href={`/embudo/ciclo/${current.data.cycle}`} onClick={event => { event.preventDefault(); navigate(`/embudo/ciclo/${current.data.cycle}`) }}><CycleSummaryCard cycle={current.data} /></a>}</section>}
      {cycle !== 'all' && visibleCycle && <section aria-label="Hitos del ciclo"><CycleMilestones milestones={visibleCycle.milestones} cycle={visibleCycle.cycle} status={visibleCycle.status} /></section>}
      <section aria-label="Resumen del embudo">{section(funnel, 'Resumen del embudo', data => <><p>{data.total} hipótesis · {data.promoted} promovidas</p><FunnelChart funnel={data} /><h2>Motivos de descarte</h2><DropReasons reasons={data.drop_reasons} /></>)}</section>
      <section aria-label="Hipótesis">{section(hypotheses, 'Hipótesis', data => data.length ? <div>{data.slice(0, funnelConfig.cardsUpTo).map(item => <div key={item.id} onClick={() => navigate(`/hipotesis/${item.id}`)}><HypothesisCard hypothesis={item} /></div>)}</div> : <p role="status">No hay hipótesis para estos filtros.</p>)}</section>
    </>}
    {cycles.status === 'error' && <ErrorState message="No pudimos cargar los ciclos disponibles." action={{ label: 'Reintentar', onClick: cycles.reload }} />}
    {cycles.status === 'ok' && cycles.data.map((item: CycleSummary) => <a key={item.cycle} href={`/embudo/ciclo/${item.cycle}`} onClick={event => { event.preventDefault(); navigate(`/embudo/ciclo/${item.cycle}`) }}>{item.cycle}</a>)}
  </main>
}
