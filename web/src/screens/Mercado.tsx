import { EtfDetail } from '../components/EtfDetail'

export function Mercado({ symbol, navigate }: { navigate: (path: string) => void; symbol?: string }) {
  if (symbol) return <EtfDetail symbol={decodeURIComponent(symbol)} navigate={navigate} />
  return <main><h1>Mercado{symbol ? ` · ${symbol}` : ''}</h1><p>En construcción</p></main>
}
