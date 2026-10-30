import { useEffect, useState } from 'react'
import { useAccounts } from './api/hooks'
import { Cuenta } from './screens/Cuenta'

function routePath() {
  return `${window.location.pathname}${window.location.search}`
}

function FirstAccount({ navigate }: { navigate: (path: string) => void }) {
  const accounts = useAccounts()
  useEffect(() => { if (accounts.status === 'ok' && accounts.data[0]) { window.history.replaceState(null, '', `/cuenta/${accounts.data[0].id}${window.location.search}`); navigate(`/cuenta/${accounts.data[0].id}${window.location.search}`) } }, [accounts.status])
  return <main>{accounts.status === 'error' ? 'No pudimos cargar las cuentas.' : 'Cargando…'}</main>
}

export function App() {
  const [path, setPath] = useState(routePath)
  useEffect(() => { const pop = () => setPath(routePath()); window.addEventListener('popstate', pop); return () => window.removeEventListener('popstate', pop) }, [])
  const navigate = (next: string) => { window.history.pushState(null, '', next); setPath(routePath()) }
  const match = path.match(/^\/cuenta(?:\/([^/?]+))?/)
  if (match) return match[1] ? <Cuenta accountId={match[1]} navigate={navigate} /> : <FirstAccount navigate={navigate} />
  return <main><h1>QuantAgent</h1><a href="/cuenta" onClick={event => { event.preventDefault(); navigate('/cuenta') }}>Cuenta</a></main>
}
