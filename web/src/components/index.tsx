import { useEffect, useRef, useState } from 'react';
import type { ElementType, HTMLAttributes, ReactNode } from 'react';
import { money, num, pct } from '../lib/format';
import './primitives.css';
import helpTexts from '../content/help-texts.es.json';

type CardProps = HTMLAttributes<HTMLElement> & { as?: 'div' | 'section' | 'article' | 'a'; href?: string; variant?: 'default' | 'dashed'; children: ReactNode };
export function Card({ as = 'div', href, variant = 'default', className = '', children, ...props }: CardProps) {
  const Tag = (href ? 'a' : as) as ElementType;
  return <Tag {...(href ? { href } : {})} className={`qa-card qa-card-${variant} ${className}`} {...props}>{children}</Tag>;
}

export function Num({ value, size = 'm', className = '', ...props }: { value: number; size?: 'xl' | 'l' | 'm' | 's'; className?: string } & HTMLAttributes<HTMLSpanElement>) {
  return <span className={`qa-num qa-num-${size} ${className}`} {...props}>{num(value)}</span>;
}

export function Delta({ value, pct: percentage, layout = 'inline', size = 'm' }: { value: number; pct?: number; layout?: 'inline' | 'stacked'; size?: 'm' | 's' }) {
  const pos = value > 0, neg = value < 0;
  const arrow = pos ? '▲' : neg ? '▼' : '—';
  const tone = pos ? 'pos' : neg ? 'neg' : 'neutral';
  const amount = `${value > 0 ? '+' : value < 0 ? '−' : ''}${money(Math.abs(value))}`;
  const label = `${pos ? 'subió' : neg ? 'bajó' : 'sin cambio'} ${num(Math.abs(value))} dólares${percentage === undefined ? '' : `, ${num(Math.abs(percentage))} por ciento`}`;
  return <span className={`qa-delta qa-${tone} qa-delta-${layout} qa-delta-${size}`} aria-label={label}><span aria-hidden="true">{arrow}</span> {amount}{percentage !== undefined && <span>{pct(percentage)}</span>}</span>;
}

export type BadgeStatus = 'filled' | 'pending' | 'partial' | 'rejected' | 'cancelled' | 'expired' | 'paper' | 'live' | 'no-data' | 'long' | 'short' | 'stage' | 'candidate' | 'production' | 'discarded' | 'soon';
const badgeText: Record<BadgeStatus, string> = { filled: 'Ejecutada', pending: 'Pendiente', partial: 'Parcial', rejected: 'Rechazada', cancelled: 'Cancelada', expired: 'Vencida', paper: 'Paper', live: 'Live · dinero real', 'no-data': 'sin datos', long: 'Larga', short: 'Corta', stage: 'Etapa', candidate: 'Candidata', production: 'Producción', discarded: 'Descartada', soon: 'Próximamente' };
const badgeIcon: Partial<Record<BadgeStatus, string>> = { filled: '✓', pending: '◷', partial: '◐', rejected: '×', cancelled: '⊘', expired: '⌛', live: '●', long: '↗', short: '↘', discarded: '×' };
export function Badge({ status, children }: { status: BadgeStatus; children?: ReactNode }) {
  return <span className={`qa-badge qa-badge-${status}`}><span aria-hidden="true">{badgeIcon[status]}</span>{children ?? badgeText[status]}</span>;
}

export function Notice({ severity, title, message, since, action }: { severity: 'critical' | 'warning' | 'info' | 'stale'; title: string; message: string; since?: string; action?: { label: string; href: string } }) {
  const icon = { critical: '⯃', warning: '△', info: 'ⓘ', stale: '◷' }[severity];
  return <div className={`qa-notice qa-notice-${severity}`} role={severity === 'critical' ? 'alert' : 'status'}><span aria-hidden="true">{icon}</span><div><strong>{title}</strong><p>{message}{since && ` · desde ${since}`}</p>{action && <a href={action.href}>{action.label}</a>}</div></div>;
}

export function Skeleton({ width = '100%', label }: { width?: string; label?: string }) {
  return <span className="qa-skeleton-wrap"><span className="qa-skeleton" style={{ width }} aria-hidden="true" />{label && <span className="qa-skeleton-label">{label}</span>}</span>;
}

function StateMessage({ icon, title, message, action, role }: { icon: string; title: string; message: string; action?: { label: string; onClick?: () => void; href?: string }; role: 'status' | 'alert' }) {
  return <section className="qa-state" role={role}><span className="qa-state-icon" aria-hidden="true">{icon}</span><h2>{title}</h2><p>{message}</p>{action && (action.href ? <a href={action.href}>{action.label}</a> : <button onClick={action.onClick}>{action.label}</button>)}</section>;
}
export function EmptyState({ title, message, action }: { title: string; message: string; action?: { label: string; onClick?: () => void; href?: string } }) {
  return <StateMessage icon="○" title={title} message={message} action={action} role="status" />;
}
export function ErrorState({ title = 'No pudimos conectar con la API', message, action }: { title?: string; message: string; action?: { label: string; onClick?: () => void; href?: string } }) {
  return <StateMessage icon="⯃" title={title} message={message} action={action} role="alert" />;
}

type HelpKey = Exclude<keyof typeof helpTexts, '_nota'>;
export function HelpTip({ term, vars = {} }: { term: HelpKey; vars?: Record<string, string> }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const entry = helpTexts[term] as { title: string; body: string; example?: string };
  const fill = (value: string) => value.replace(/\{([^}]+)\}/g, (_, key: string) => vars[key] ?? `{${key}}`);
  useEffect(() => {
    if (!open) return;
    const keydown = (event: KeyboardEvent) => { if (event.key === 'Escape') { setOpen(false); button.current?.focus(); } };
    const outside = (event: MouseEvent) => { if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false); };
    document.addEventListener('keydown', keydown); document.addEventListener('mousedown', outside);
    return () => { document.removeEventListener('keydown', keydown); document.removeEventListener('mousedown', outside); };
  }, [open]);
  return <span className="qa-help" ref={root}><button ref={button} type="button" className="qa-help-trigger" aria-label={`Qué es ${entry.title}`} aria-expanded={open} aria-controls={`qa-help-${term}`} onClick={() => setOpen(!open)}>?</button>{open && <span className="qa-help-panel" id={`qa-help-${term}`} role="dialog" aria-label={entry.title}><strong>{entry.title}</strong><span>{entry.body}</span>{entry.example && <span>{fill(entry.example)}</span>}<button type="button" className="qa-help-close" aria-label="Cerrar" onClick={() => { setOpen(false); button.current?.focus(); }}>Cerrar</button></span>}</span>;
}

export function ComingSoon({ title, children, example = false }: { title: string; children?: ReactNode; example?: boolean }) {
  return <section className={`qa-coming-soon${example ? ' qa-coming-soon-example' : ''}`} aria-label={`${title}, próximamente`}><header><h2>{title}</h2><Badge status="soon" /></header>{children ?? (example ? <div className="qa-coming-example" aria-label="Datos de ejemplo"><Skeleton width="70%" /><Skeleton width="42%" /></div> : <div className="qa-coming-ghost" aria-hidden="true" />)}</section>;
}

export function SegmentedControl({ label, options, value, onChange, disabled = false }: { label: string; options: { value: string; label: string; disabled?: boolean }[]; value: string; onChange: (value: string) => void; disabled?: boolean }) {
  return <div className="qa-segmented" role="radiogroup" aria-label={label}>{options.map(option => <button key={option.value} type="button" role="radio" aria-checked={value === option.value} disabled={disabled || option.disabled} className="qa-segment" onClick={() => onChange(option.value)}>{option.label}</button>)}</div>;
}

export type ChipOption = { value: string; label: string; count?: number };
export function Chip({ label, count, selected = false, disabled = false, onClick }: { label: string; count?: number; selected?: boolean; disabled?: boolean; onClick?: () => void }) {
  const unavailable = disabled || count === 0;
  return <button type="button" className="qa-chip" aria-pressed={selected} disabled={unavailable} onClick={onClick}><span aria-hidden="true">{selected ? '✓' : ''}</span>{label}{count !== undefined && <span className="qa-chip-count">{count}</span>}</button>;
}
export function ChipGroup({ label, options, selected, onChange }: { label: string; options: ChipOption[]; selected: string[]; onChange: (selected: string[]) => void }) {
  const toggle = (value: string) => onChange(selected.includes(value) ? selected.filter(item => item !== value) : [...selected, value]);
  return <div className="qa-chip-group" role="group" aria-label={label}>{options.map(option => <Chip key={option.value} {...option} selected={selected.includes(option.value)} onClick={() => toggle(option.value)} />)}</div>;
}
