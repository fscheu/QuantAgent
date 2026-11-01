import { useEffect, useMemo, useState } from 'react'
import type { OrderStatus } from '../api/types'
import { useAccounts, useDividends, useEquity, useOrders, usePositions, useSectorExposure, useSummary } from '../api/hooks'
import { dataConfig, featureFlags } from '../config'
import { money, relative, time } from '../lib/format'
import { AccountNoData, AccountSelect, AccountSummaryCard, ChipGroup, DataTable, DateStepper, DivergingBar, EmptyState, EquityChart, ErrorState, Notice, OrderCard, orderColumns, orderGroupBy, orderSubRow, PositionCard, positionColumns, Skeleton } from '../components'
import type { EquityRange } from '../components'
import './Cuenta.css'

const todayNY = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date())
type Filter = 'all' | OrderStatus
const filters: { value: Filter; label: string; statuses: OrderStatus[] }[] = [
  { value: 'all', label: 'Todas', statuses: [] }, { value: 'rejected', label: 'Rechazadas', statuses: ['rejected'] },
  { value: 'pending', label: 'Pendientes', statuses: ['pending', 'partially_filled'] }, { value: 'filled', label: 'Ejecutadas', statuses: ['filled'] },
  { value: 'cancelled', label: 'Canceladas', statuses: ['cancelled'] }, { value: 'expired', label: 'Vencidas', statuses: ['expired'] },
]

export function Cuenta({ accountId, navigate }: { accountId: string; navigate: (path: string) => void }) {
  const accounts = useAccounts()
  const summary = useSummary(accountId)
  const positions = usePositions(accountId)
  const [range, setRange] = useState<EquityRange>('1M')
  const equity = useEquity(accountId, range)
  const sectorExposure = useSectorExposure(accountId)
  const dividends = useDividends(accountId)
  const [date, setDate] = useState(() => new URLSearchParams(location.search).get('date') || todayNY())
  const [filter, setFilter] = useState<Filter>(() => (new URLSearchParams(location.search).get('estado') as Filter) || 'all')
  const orders = useOrders(accountId, date)
  useEffect(() => { const params = new URLSearchParams(location.search); params.set('date', date); filter === 'all' ? params.delete('estado') : params.set('estado', filter); history.replaceState(null, '', `${location.pathname}?${params}`) }, [date, filter])
  const account = summary.status === 'ok' ? summary.data : null
  const orderCounts = useMemo(() => filters.map(item => ({ value: item.value, label: item.label, count: item.value === 'all' ? (orders.status === 'ok' ? orders.data.length : 0) : orders.status === 'ok' ? orders.data.filter(o => item.statuses.includes(o.status)).length : 0 })), [orders])
  if (account?.status === 'no_data') return <AccountNoData account={account} otherAccounts={accounts.status === 'ok' ? accounts.data.filter(a => a.id !== account.id && a.status === 'active').map(a => ({ id: a.id, name: a.name, href: `/cuenta/${a.id}` })) : []} />
  const visibleOrders = orders.status === 'ok' ? orders.data.filter(o => filter === 'all' || filters.find(f => f.value === filter)?.statuses.includes(o.status)) : []
  const staleMinutes = dataConfig.staleMinutesMarketOpen
  const isStale = account?.status === 'active' && (account.stale || Date.now() - new Date(account.as_of).getTime() > staleMinutes * 60000)
  const changeDate = (next: string) => setDate(next)
  return <main className="qa-cuenta">
    {featureFlags.accountPerformanceTab && <nav role="tablist" aria-label="Secciones de cuenta" className="qa-account-tabs"><a role="tab" aria-selected="true" href={`/cuenta/${accountId}`} onClick={event => { event.preventDefault(); navigate(`/cuenta/${accountId}`) }}>Hoy</a><a role="tab" aria-selected="false" href={`/cuenta/${accountId}/rendimiento`} onClick={event => { event.preventDefault(); navigate(`/cuenta/${accountId}/rendimiento`) }}>Rendimiento</a></nav>}
    <header className="qa-cuenta-top"><div>{accounts.status === 'ok' ? <AccountSelect accounts={accounts.data} value={accountId} onChange={id => navigate(`/cuenta/${id}`)} /> : accounts.status === 'error' ? <ErrorState message="No pudimos cargar las cuentas." action={{ label: 'Reintentar', onClick: accounts.reload }} /> : <Skeleton width="220px" label="Cuentas" />}</div><p>Actualizado {account?.status === 'active' ? time(account.as_of, 'ART') : '—'}{account?.status === 'active' ? ` · ${relative(account.as_of)}` : ''}</p></header>
    {isStale && <Notice severity="stale" title="Datos desactualizados" message={`Sin datos nuevos desde hace más de ${staleMinutes} min.`} />}
    <section><h2>Resumen</h2>{summary.status === 'loading' ? <Skeleton label="Resumen" /> : summary.status === 'error' ? <ErrorState message="No pudimos cargar el resumen." action={{ label: 'Reintentar', onClick: summary.reload }} /> : <AccountSummaryCard account={summary.data} />}</section>
    {account?.status === 'active' && account.alarms.map(alarm => <Notice key={alarm.id} severity={alarm.severity} title={alarm.title} message={alarm.message} since={time(alarm.since, 'ET')} />)}
    <section><h2>Posiciones <span className="qa-section-help">Largas y cortas</span></h2>{positions.status === 'loading' ? <Skeleton label="Posiciones" /> : positions.status === 'error' ? <ErrorState message="No pudimos cargar las posiciones." action={{ label: 'Reintentar', onClick: positions.reload }} /> : positions.data.length === 0 ? <EmptyState title="Sin posiciones" message="No hay posiciones abiertas en esta cuenta." /> : <><div className="qa-cuenta-desktop"><DataTable columns={positionColumns} rows={positions.data} rowKey={p => p.symbol} caption="Posiciones" state="ready" /></div><div className="qa-cuenta-mobile">{positions.data.map(position => <PositionCard key={position.symbol} position={position} />)}</div></>}</section>
    <section><header className="qa-cuenta-orders-heading"><h2>Órdenes</h2><DateStepper value={date} onChange={changeDate} /></header><ChipGroup label="Filtrar órdenes" options={orderCounts.map(x => ({ ...x }))} selected={[filter]} onChange={selected => setFilter((selected.at(-1) as Filter) ?? 'all')} />{orders.status === 'loading' ? <Skeleton label="Órdenes" /> : orders.status === 'error' ? <ErrorState message="No pudimos cargar las órdenes." action={{ label: 'Reintentar', onClick: orders.reload }} /> : visibleOrders.length === 0 ? <EmptyState title={orders.data.length === 0 ? 'Sin órdenes ese día' : 'Sin órdenes con este estado'} message={orders.data.length === 0 ? 'No se registraron órdenes para esta fecha.' : 'No hay órdenes con este estado.'} /> : <><div className="qa-cuenta-desktop"><DataTable columns={orderColumns} rows={[...visibleOrders].sort((a, b) => Number(b.status === 'rejected') - Number(a.status === 'rejected'))} rowKey={o => o.id} caption="Órdenes" state="ready" groupBy={orderGroupBy} subRow={orderSubRow} /></div><div className="qa-cuenta-mobile">{[...visibleOrders].sort((a,b) => Number(b.status === 'rejected') - Number(a.status === 'rejected')).map(order => <OrderCard key={order.id} order={order} />)}</div></>}</section>
    <section><h2>Curva de valor</h2>{equity.status === 'loading' ? <Skeleton label="Curva de valor" /> : equity.status === 'error' ? <ErrorState message="No pudimos cargar la curva de valor." action={{ label: 'Reintentar', onClick: equity.reload }} /> : <EquityChart data={equity.data} range={range} onRangeChange={setRange} />}</section>
    <section><h2>Exposición por sector</h2>{sectorExposure.status === 'loading' ? <Skeleton label="Exposición por sector" /> : sectorExposure.status === 'error' ? <ErrorState message="No pudimos cargar la exposición por sector." action={{ label: 'Reintentar', onClick: sectorExposure.reload }} /> : <DivergingBar data={sectorExposure.data} />}</section>
    <section><h2>Dividendos</h2>{dividends.status === 'loading' ? <Skeleton label="Dividendos" /> : dividends.status === 'error' ? <ErrorState message="No pudimos cargar los dividendos." action={{ label: 'Reintentar', onClick: dividends.reload }} /> : dividends.data.length === 0 ? <EmptyState title="Sin dividendos" message="No se registraron dividendos para esta cuenta." /> : <div className="qa-dividends" role="table" aria-label="Dividendos"><div role="row" className="qa-dividends-head"><span role="columnheader">Fecha</span><span role="columnheader">Símbolos</span><span role="columnheader">Neto</span><span role="columnheader">Retenido</span></div>{dividends.data.map(item => <div role="row" className="qa-dividends-row" key={item.date + item.symbols.join('-')}><span role="cell">{item.date}</span><span role="cell">{item.symbols.join(', ')}</span><span role="cell">{money(item.net)}</span><span role="cell">{money(item.withheld)}</span></div>)}</div>}</section>
  </main>
}
