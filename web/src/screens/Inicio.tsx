import { useMemo } from 'react'
import type { CalendarEvent, FeedEvent, HomeStatus } from '../api/types'
import { useAccounts, useCalendar, useFeed, useHomeStatus } from '../api/hooks'
import { homeConfig } from '../config'
import { AccountCard } from '../components/Account'
import { ComingSoon, ErrorState, Notice, Skeleton } from '../components'
import { time } from '../lib/format'
import './Inicio.css'

const severityRank = (severity: FeedEvent['severity']) => homeConfig.feedSeverityOrder.indexOf(severity as typeof homeConfig.feedSeverityOrder[number])
const sortFeed = (events: FeedEvent[]) => [...events].sort((a, b) => severityRank(a.severity) - severityRank(b.severity) || Date.parse(b.occurred_at) - Date.parse(a.occurred_at))

function HomeStatusBanner({ status, reload }: { status: HomeStatus | null; reload: () => void }) {
  if (!status) return <Skeleton label="Estado general" />
  const severity = status.level === 'ok' || status.level === 'neutral' ? 'info' : status.level
  return <div className="qa-home-banner">{status.link ? <a href={status.link} className="qa-home-status-link"><Notice severity={severity} title={status.headline} message={status.message} /></a> : <Notice severity={severity} title={status.headline} message={status.message} />}{!status.link && <button className="qa-home-reload" onClick={reload}>Actualizar estado</button>}</div>
}

function FeedList({ events, reload }: { events: FeedEvent[] | null; reload: () => void }) {
  if (!events) return <Skeleton label="Actividad" />
  if (events.length === 0) return <p className="qa-home-muted">Nada nuevo desde {time(new Date(Date.now() - homeConfig.feedFallbackHours * 3600000).toISOString(), 'ART')}.</p>
  const visible = sortFeed(events).slice(0, homeConfig.feedMaxItems)
  return <><ul className="qa-home-feed">{visible.map(event => <li key={event.id}><a href={event.link}><span className={`qa-home-glyph qa-home-glyph-${event.severity}`} aria-hidden="true">{event.severity === 'critical' ? '⯃' : event.severity === 'warning' ? '△' : event.type === 'cycle_completed' ? '➜' : event.severity === 'neutral' ? '+' : 'ⓘ'}</span><span><strong>{event.title}{event.group_count > 1 ? ` · ${event.group_count}` : ''}</strong>{event.subtitle && <small>{event.subtitle}</small>}<small>{time(event.occurred_at, 'ART')}</small></span></a></li>)}</ul>{events.length > homeConfig.feedMaxItems && <button className="qa-home-reload" onClick={reload}>Ver todo</button>}</>
}

function EventList({ events }: { events: CalendarEvent[] | null }) {
  if (!events) return <Skeleton label="Próximos eventos" />
  if (!events.length) return <p className="qa-home-muted">No hay eventos próximos.</p>
  return <ul className="qa-home-events">{[...events].sort((a, b) => a.date.localeCompare(b.date) || (a.starts_at ?? '').localeCompare(b.starts_at ?? '')).map(event => <li key={event.id}><time dateTime={event.date}>{new Intl.DateTimeFormat('es-AR', { timeZone: 'UTC', day: '2-digit', month: 'short' }).format(new Date(`${event.date}T12:00:00Z`))}</time><span><strong>{event.title}</strong>{event.subtitle && <small>{event.subtitle}</small>}</span></li>)}</ul>
}

export function Inicio({ navigate }: { navigate: (path: string) => void }) {
  const status = useHomeStatus(), accounts = useAccounts(), feed = useFeed(), calendar = useCalendar(homeConfig.calendarDays)
  const sortedFeed = useMemo(() => feed.status === 'ok' ? sortFeed(feed.data) : null, [feed])
  return <main className="qa-inicio"><header className="qa-inicio-header"><h1>Inicio</h1><nav aria-label="Navegación"><a href="/" aria-current="page">Inicio</a><a href="/cuenta" onClick={event => { event.preventDefault(); navigate('/cuenta') }}>Cuenta</a></nav></header>
    <section className="qa-home-status" aria-label="Estado general">{status.status === 'error' ? <ErrorState message="No pudimos cargar el estado general." action={{ label: 'Reintentar', onClick: status.reload }} /> : <HomeStatusBanner status={status.status === 'ok' ? status.data : null} reload={status.reload} />}</section>
    <section className="qa-home-accounts"><h2>Cuentas</h2>{accounts.status === 'loading' ? <Skeleton label="Cuentas" /> : accounts.status === 'error' ? <ErrorState message="No pudimos cargar las cuentas." action={{ label: 'Reintentar', onClick: accounts.reload }} /> : <div className="qa-home-account-grid">{accounts.data.map(account => <AccountCard key={account.id} account={account} href={`/cuenta/${account.id}`} />)}</div>}</section>
    <div className="qa-home-panels"><section><h2>Desde tu última visita</h2>{feed.status === 'error' ? <ErrorState message="No pudimos cargar la actividad." action={{ label: 'Reintentar', onClick: feed.reload }} /> : <FeedList events={sortedFeed} reload={feed.reload} />}</section><ComingSoon title="Embudo" /><section><h2>Próximos eventos</h2>{calendar.status === 'error' ? <ErrorState message="No pudimos cargar los eventos." action={{ label: 'Reintentar', onClick: calendar.reload }} /> : <EventList events={calendar.status === 'ok' ? calendar.data : null} />}</section></div>
  </main>
}
