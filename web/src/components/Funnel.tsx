import type { CycleMilestone, ReviewRule, Verdict } from '../api/types';
import { cycleConfig } from '../config';
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
