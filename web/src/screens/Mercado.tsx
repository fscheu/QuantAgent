export function Mercado({ symbol }: { navigate: (path: string) => void; symbol?: string }) {
  return <main><h1>Mercado{symbol ? ` · ${symbol}` : ''}</h1><p>En construcción</p></main>
}
