import type { CycleMilestone, DropReason, Funnel, Measure, ReviewRule, Verdict } from '../api/types';
import { cycleConfig, funnelConfig } from '../config';
import './Funnel.css';

const verdicts: Record<Verdict, { icon: string; className: string }> = {
  PASA: { icon: '✓', className: 'qa-seg-passed' },
  CORTA: { icon: '×', className: 'qa-seg-cut' },
  ERROR: { icon: '!', className: 'qa-seg-error' },
  ALARMA: { icon: '♧', className: 'qa-seg-alarm' },
};

export function StageVerdictBadge({ verdict, promoted = false, size = 'md' }: { verdict: Verdict; promoted?: boolean; size?: 'md' | 'sm' }) {
  const { icon, className } = verdicts[verdict];
  return <span className={`qa-stage-verdict ${className} qa-stage-verdict-${size}`} aria-label={promoted ? `${verdict}, promovida` : verdict}>
    <span aria-hidden="true">{promoted ? '★' : icon}</span>{verdict}{promoted && <span className="qa-stage-promoted">PROMOVIDA</span>}
  </span>;
}

export function ReviewCounter({ rule, loading = false, error = false }: { rule: ReviewRule | null; loading?: boolean; error?: boolean }) {
  if (loading) return <section className="qa-review-counter" role="status">Cargando racha de ciclos cerrados…</section>;
  if (error) return <section className="qa-review-counter" role="alert">No se pudo cargar la regla de revisión.</section>;
  if (!rule || rule.last_cycles.length === 0) return <section className="qa-review-counter" role="status">Sin datos: todavía no hay ciclos cerrados para contar.</section>;
  const limit = cycleConfig.reviewAfterCycles;
  return <section className={`qa-review-counter${rule.streak >= limit ? ' qa-review-counter-reached' : ''}`} aria-label={`${rule.streak} de ${limit} ciclos seguidos sin promovida${rule.streak >= limit ? ', alcanzada' : ''}`}>
    <strong>{rule.streak} de {limit} ciclos seguidos sin promovida{rule.streak >= limit && ' · alcanzada'}</strong>
    <div className="qa-review-cycle-track" aria-hidden="true">{rule.last_cycles.slice(-limit).map(({ cycle, promoted }) => <span key={cycle} className={promoted > 0 ? 'qa-review-promoted' : ''}>{promoted > 0 ? '★' : '×'}</span>)}</div>
    <p>{rule.last_cycles.map(({ cycle, promoted }) => `${cycle.slice(5)}: ${promoted > 0 ? 'con promovida' : 'sin promovida'}`).join(' · ')}</p>
  </section>;
}

const milestoneLabels: Record<CycleMilestone['key'], string> = { proposal: 'Propuesta', funnel_start: 'Arranque del embudo', paper_review: 'Revisión de paper', close: 'Cierre' };
export function CycleMilestones({ milestones, cycle, status }: { milestones: CycleMilestone[] | null; cycle: string; status: 'open' | 'closed' }) {
  const [year, month] = cycle.split('-').map(Number);
  const items = milestones ?? (Object.entries(cycleConfig.milestones).map(([key, day]) => ({ key: key === 'funnelStart' ? 'funnel_start' : key === 'paperReview' ? 'paper_review' : 'proposal', day, date: new Date(Date.UTC(year, month - 1, day)).toISOString(), done: false })) as CycleMilestone[]);
  const now = new Date();
  return <ol className="qa-cycle-milestones" aria-label={`Hitos del ciclo ${cycle}`}>{items.map((item, index) => {
    const date = new Date(item.date);
    const overdue = !item.done && date < now;
    const label = milestoneLabels[item.key];
    const state = item.done ? 'hecho' : status === 'closed' ? 'cerrado' : overdue ? 'vencido' : 'pendiente';
    return <li key={`${item.key}-${index}`} className={item.done ? 'is-done' : overdue ? 'is-overdue' : ''}><span className="qa-milestone-dot" aria-hidden="true">{item.done ? '✓' : overdue ? '!' : '·'}</span><span><strong>{label}</strong><small>{date.toLocaleDateString('es-AR', { weekday: 'long', day: '2-digit', month: 'short', timeZone: 'UTC' })}</small></span><span className="qa-milestone-state">{state}</span></li>;
  })}</ol>;
}

function formatMeasure(measure: Measure) {
  const number = (value: number | null) => value === null ? '—' : new Intl.NumberFormat('es-AR', { maximumFractionDigits: 2 }).format(value);
  const suffix = measure.unit === 'pct' ? '%' : measure.unit === 'years' ? ' años' : '';
  return `${number(measure.value)}${suffix} · umbral ${measure.comparator ?? ''} ${number(measure.threshold)}${suffix}`;
}

export function FunnelChart({ funnel }: { funnel: Funnel }) {
  if (funnel.sample_too_small || funnel.total < funnelConfig.minForChart) return <section className="qa-funnel-empty" role="status">Muestra chica: {funnel.total} hipótesis; se necesitan {funnelConfig.minForChart} para mostrar el embudo.</section>;
  return <section className="qa-funnel-chart" aria-label="Embudo de hipótesis">
    {funnelConfig.stages.map(({ id, code, label }, index) => {
      const stage = funnel.stages.find(item => item.stage === id);
      if (!stage) return null;
      const segments = [{ key: 'passed', label: 'PASA', count: stage.passed }, { key: 'cut', label: 'CORTA', count: stage.cut }, { key: 'error', label: 'ERROR', count: stage.error }, { key: 'alarm', label: 'ALARMA', count: stage.alarm }];
      return <div className="qa-funnel-row" key={id}><div className="qa-funnel-row-heading"><strong>{code} · {label}</strong><span>llegaron {stage.reached} · acá {stage.here} · cayeron {stage.cut}</span></div>
        {stage.reached ? <div className="qa-funnel-bar" role="img" aria-label={`${code}: ${segments.map(({ label: verdict, count }) => `${verdict} ${count}`).join(', ')}`}>
          {segments.filter(({ count }) => count > 0).map(({ key, label: verdict, count }) => <span key={key} className={`qa-funnel-segment qa-seg-${key}`} style={{ width: `${count / stage.reached * 100}%` }} aria-hidden="true">{verdict} {count}</span>)}
        </div> : <p className="qa-funnel-not-reached">·········· Todavía ninguna</p>}
        {index === 3 && funnel.promoted > 0 && <small>{funnel.promoted} promovidas</small>}
      </div>;
    })}
    {funnel.stages.reduce((sum, stage) => sum + stage.error, 0) > 0 && <p>Detenidas aparte: error {funnel.stages.reduce((sum, stage) => sum + stage.error, 0)}</p>}
    <div className="qa-funnel-later" aria-label="Fases posteriores">{funnelConfig.laterPhases.map(({ id, label }) => { const phase = funnel.later.find(item => item.phase === id); return <span key={id}>{label}: {phase?.reached ?? 0} · acá {phase?.here ?? 0}</span>; })}</div>
  </section>;
}

export function DropReasons({ reasons }: { reasons: DropReason[] }) {
  if (!reasons.length) return <p className="qa-drop-empty" role="status">Ninguna cayó en esta etapa.</p>;
  return <ul className="qa-drop-reasons">{reasons.map((reason, index) => <li key={`${reason.stage}-${reason.cause_code}-${index}`}><div><strong>{reason.cause_label}</strong><span>{reason.stage.toUpperCase()} · {reason.count}</span></div><StageVerdictBadge verdict={reason.verdict} size="sm" />{reason.example && <code>{reason.example.metric}: {formatMeasure(reason.example)}</code>}</li>)}</ul>;
}
