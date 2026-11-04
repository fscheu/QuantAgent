import './AppNav.css'

const destinations = [
  { label: 'Inicio', href: '/', icon: '⌂', match: (path: string) => path === '/' },
  { label: 'Cuenta', href: '/cuenta', icon: '◉', match: (path: string) => path.startsWith('/cuenta') },
  { label: 'Backtests', href: '/backtests', icon: '◷', match: (path: string) => path === '/backtests' || path.startsWith('/rendimiento/backtest/') },
  { label: 'Embudo', href: '/embudo', icon: '▽', match: (path: string) => path.startsWith('/embudo') || path.startsWith('/hipotesis') },
  { label: 'Mercado', href: '/mercado', icon: '⌁', match: (path: string) => path.startsWith('/mercado') },
]

function Icon({ name }: { name: string }) {
  const paths: Record<string, string> = {
    '⌂': '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    '◉': '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    '◷': '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5m4-1v5l3 2"/>',
    '▽': '<path d="m4 5 8 14 8-14z"/>',
    '⌁': '<path d="M3 17c4-9 6 9 10 0s6 0 8-5"/><path d="M3 7h18"/>',
  }
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" dangerouslySetInnerHTML={{ __html: paths[name] }} />
}

export function AppNav({ path, navigate }: { path: string; navigate: (path: string) => void }) {
  return <nav className="qa-app-nav" aria-label="Secciones">
    {destinations.map(destination => {
      const active = destination.match?.(path.split('?')[0])
      const content = <><Icon name={destination.icon} /><span className="qa-app-nav-label">{destination.label}</span></>
      return destination.href
        ? <a key={destination.label} href={destination.href} className={active ? 'is-active' : ''} aria-current={active ? 'page' : undefined} onClick={event => { event.preventDefault(); navigate(destination.href!) }}>{content}</a>
        : <span key={destination.label} className="qa-app-nav-disabled" aria-disabled="true">{content}</span>
    })}
  </nav>
}
