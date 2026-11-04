export function Embudo({ ciclo }: { navigate: (path: string) => void; ciclo?: string }) {
  return <main><h1>Embudo{ciclo ? ` · ${ciclo}` : ''}</h1><p>En construcción</p></main>
}
