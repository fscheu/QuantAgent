import type { CycleReceipt as ReceiptData, Measure } from '../api/types'
import { StageVerdictBadge } from './Funnel'
import './CycleReceipt.css'

const measureText = (m: Measure) => `${m.value ?? '—'} · umbral ${m.comparator ?? ''} ${m.threshold ?? '—'}`
const countLabels = { entered: 'Entraron', active: 'Siguen', cut: 'Cayeron', held: 'Detenidas', promoted: 'Promovidas' }
export function CycleReceipt({ receipt, presentation = 'screen', onClose }: { receipt: ReceiptData; presentation?: 'screen' | 'panel'; onClose: () => void }) {
  return <main className={`qa-cycle-receipt ${presentation}`} role={presentation === 'panel' ? 'dialog' : undefined} aria-modal={presentation === 'panel' || undefined} aria-label={`Recibo ${receipt.cycle}`}>
    <header><button onClick={onClose} aria-label="Cerrar recibo">×</button><h1>Recibo · {receipt.cycle}</h1>{receipt.partial ? <p>parcial · todavía abierto</p> : <code>{receipt.hash?.slice(0, 12)}</code>}</header>
    <dl>{Object.entries(receipt.counts).map(([key, value]) => <div key={key}><dt>{countLabels[key as keyof typeof countLabels]}</dt><dd>{value}</dd></div>)}</dl>
    <h2>Etapas</h2><table><thead><tr><th>Etapa</th><th>Llegaron</th><th>PASA</th><th>CORTA</th><th>ERROR</th><th>ALARMA</th></tr></thead><tbody>{receipt.stages.map(s => <tr key={s.stage}><th>{s.stage.toUpperCase()}</th>{[s.reached, s.passed, s.cut, s.error, s.alarm].map((count, i) => <td key={i}>{s.pending ? 'pendiente' : count}</td>)}</tr>)}</tbody></table>
    <h2>Descartes y detenidas</h2>{receipt.drops.map((drop, i) => <p key={`${drop.hypothesis_id}-${i}`}>{drop.name} · {drop.stage.toUpperCase()} <StageVerdictBadge verdict={drop.verdict} size="sm" /> · {drop.reason}{drop.measure && <> · {measureText(drop.measure)}</>}</p>)}
    <h2>Intentos por hipótesis</h2>{receipt.attempts.map(a => <p key={a.hypothesis_id}>{a.name} · {a.valid} válidos / {a.invalid} inválidos</p>)}
    <h2>Datos usados</h2>{receipt.snapshots.map(s => <p key={s.name}>{s.name} · {s.as_of || '—'} · <code>{s.hash.slice(0, 12)}</code></p>)}
  </main>
}
