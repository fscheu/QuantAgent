import type { Attempt, HypothesisDetail, StageResult } from '../api/types'
import { funnelConfig } from '../config'
import { StageVerdictBadge } from './Funnel'
import './HypothesisDetail.css'

const date = (value: string) => new Date(value).toLocaleDateString('es-AR', { year: 'numeric', month: 'short', day: '2-digit', timeZone: 'UTC' })
const measure = (item: NonNullable<StageResult['measure']>) => {
  const suffix = item.unit === 'pct' ? '%' : item.unit === 'years' ? ' años' : ''
  return `${item.metric}: ${item.value ?? '—'}${suffix} · umbral ${item.comparator ?? ''} ${item.threshold ?? '—'}${suffix}`
}

export function StageTimeline({ timeline }: { timeline: StageResult[] }) {
  const stages = funnelConfig.stages.map(({ id, code, label }) => ({ id, code, label, result: timeline.find(item => item.stage === id) }))
  const firstPending = stages.findIndex(item => !item.result)
  return <section className="qa-hypothesis-section"><h2>Etapas</h2><ol className="qa-stage-timeline">{stages.map(({ id, code, label, result }, index) => <li key={id} className={`${result ? '' : 'is-pending'}${firstPending >= 0 && index > firstPending ? ' is-later' : ''}`}><strong>{code} · {label}</strong>{result ? <><StageVerdictBadge verdict={result.verdict} size="sm" /><time dateTime={result.decided_at}>{date(result.decided_at)}</time><p>{result.reason}</p>{result.measure && <small>{measure(result.measure)}</small>}{result.imported && <span className="qa-imported">Importado</span>}</> : <span>No llegó</span>}</li>)}</ol></section>
}

export function AttemptList({ attempts }: { attempts: Attempt[] }) {
  if (!attempts.length) return <section className="qa-hypothesis-section"><h2>Intentos</h2><p role="status">Sin intentos.</p></section>
  return <section className="qa-hypothesis-section"><h2>Intentos</h2><ul className="qa-attempt-list">{attempts.map(item => <li key={item.id} className={item.invalid ? 'is-invalid' : ''}><time dateTime={item.ran_at}>{date(item.ran_at)}</time><strong>{item.subject}</strong><span>Snapshot {item.snapshot_hash}</span><span>Sharpe {item.sharpe ?? '—'} · Operaciones {item.trades ?? '—'} · PnL {item.pnl_pct === null ? '—' : `${item.pnl_pct}%`}</span><span>{item.stage.toUpperCase()}</span>{item.invalid && <p>Inválido{item.invalid_reason ? `: ${item.invalid_reason}` : ''}</p>}</li>)}</ul></section>
}

export function HypothesisDetailContent({ hypothesis, navigate }: { hypothesis: HypothesisDetail; navigate: (path: string) => void }) {
  const phaseLabels = Object.fromEntries(funnelConfig.laterPhases.map(item => [item.id, item.label]))
  return <><header><h1>{hypothesis.name}</h1>{hypothesis.verdict && <StageVerdictBadge verdict={hypothesis.verdict} promoted={hypothesis.promoted} />}</header>{hypothesis.idea && <p>{hypothesis.idea}</p>}{hypothesis.cycle && <p>Ciclo: <a href={`/embudo/ciclo/${hypothesis.cycle}`} onClick={event => { event.preventDefault(); navigate(`/embudo/ciclo/${hypothesis.cycle}`) }}>{hypothesis.cycle}</a></p>}{hypothesis.used_reserve && <p className="qa-reserve-chip">Usó la reserva 2023+</p>}{hypothesis.paper_account && <p>Cuenta paper: {hypothesis.paper_account.name}</p>}<StageTimeline timeline={hypothesis.timeline} />{hypothesis.later_phases.length > 0 && <section className="qa-hypothesis-section qa-later-phases"><h2>Fases posteriores</h2><ul>{hypothesis.later_phases.map((phase, index) => <li key={`${phase.phase}-${index}`}>{phaseLabels[phase.phase]} · {date(phase.since)}{phase.until && ` – ${date(phase.until)}`}</li>)}</ul></section>}<AttemptList attempts={hypothesis.attempts} /><section className="qa-hypothesis-section"><h2>Backtests vinculados</h2>{hypothesis.backtests.length ? <ul>{hypothesis.backtests.map(item => <li key={item.id}>{item.label}{item.stage && ` · ${item.stage.toUpperCase()}`}</li>)}</ul> : <p>Sin backtests vinculados todavía</p>}</section></>
}
