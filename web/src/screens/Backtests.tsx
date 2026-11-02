import { useState } from 'react'
import { useBacktests } from '../api/hooks'
import { ErrorState, Skeleton } from '../components'

export function Backtests({ navigate }: { navigate:(path:string)=>void }) {
 const runs=useBacktests(); const [selected,setSelected]=useState<string[]>([])
 const toggle=(id:string)=>setSelected(current=>current.includes(id)?current.filter(x=>x!==id):current.length<2?[...current,id]:current)
 return <main className="qa-performance"><header className="qa-perf-head"><h1>Backtests</h1><p>Corridas históricas y su rendimiento.</p></header>
 {runs.status==='loading'?<Skeleton label="Backtests"/>:runs.status==='error'?<ErrorState message="No pudimos cargar los backtests." action={{label:'Reintentar',onClick:runs.reload}}/>:runs.data.length===0?<p>No hay backtests disponibles.</p>:<><div className="qa-backtests">{runs.data.map(run=><article key={run.id}><label><input type="checkbox" checked={selected.includes(run.id)} disabled={!selected.includes(run.id)&&selected.length===2} onChange={()=>toggle(run.id)} aria-label={`Seleccionar ${run.label}`}/><span><strong>{run.label}</strong><small>{run.hypothesis_name??'Sin hipótesis'} · {run.start} – {run.end}</small></span><b>{run.total_return.toFixed(1)} %</b></label><button onClick={()=>navigate(`/rendimiento/backtest/${run.id}`)}>Ver rendimiento</button></article>)}</div>{selected.length===2&&<button className="qa-primary" onClick={()=>navigate(`/rendimiento/backtest/${selected[0]}?vs=backtest:${selected[1]}`)}>Comparar 2 ›</button>}</>}
 </main>
}
