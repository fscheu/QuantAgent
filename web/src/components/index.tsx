import type { ElementType, HTMLAttributes, ReactNode } from 'react';
import { money, num, pct } from '../lib/format';
import './primitives.css';

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
