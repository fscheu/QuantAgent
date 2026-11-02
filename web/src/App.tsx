import { useEffect, useState, type ReactNode } from 'react'
import { useAccounts } from './api/hooks'
import { Cuenta } from './screens/Cuenta'
import { Inicio } from './screens/Inicio'
import { Backtests } from './screens/Backtests'
import { Performance } from './screens/Performance'
import { AppNav } from './components/AppNav'

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
  const perf = path.match(/^\/rendimiento\/backtest\/([^/?]+)/)
  let screen: ReactNode
  if (perf) screen = <Performance id={perf[1]} navigate={navigate} />
  const runPerf = path.match(/^\/rendimiento\/run\/([^/?]+)/)
  if (runPerf) screen = <Performance id={runPerf[1]} type="run" navigate={navigate} />
  const accountPerf = path.match(/^\/cuenta\/([^/?]+)\/rendimiento/)
  if (accountPerf) screen = <Performance id={accountPerf[1]} type="portfolio" navigate={navigate} />
  if (!screen && path.startsWith('/backtests')) screen = <Backtests navigate={navigate} />
  const match = path.match(/^\/cuenta(?:\/([^/?]+))?/)
  if (!screen && match) screen = match[1] ? <Cuenta accountId={match[1]} navigate={navigate} /> : <FirstAccount navigate={navigate} />
  if (!screen) screen = <Inicio navigate={navigate} />
  return <div className="qa-app-shell"><AppNav path={path} navigate={navigate} /><div className="qa-app-content">{screen}</div></div>
}
