import type { AccountSummary, AccountSummaryNoData } from '../api/types';
import { money, time } from '../lib/format';
import { Badge, Card, Delta, Num } from './index';
import { Sparkline } from './Charts';
import './Account.css';

function AccountHeading({ account }: { account: AccountSummary }) {
  return <header className="qa-account-heading"><h3>{account.name}</h3><Badge status={account.status === 'no_data' ? 'no-data' : account.env} /></header>;
}

export function AccountCard({ account, href }: { account: AccountSummary; href?: string }) {
  if (account.status === 'no_data') return <Card as="article" href={href} variant="dashed" className="qa-account-card qa-account-no-data-card"><AccountHeading account={account} /><strong>Sin datos todavía</strong><p>{noDataReason(account.no_data_reason)}</p></Card>;
  return <Card as="article" href={href} className="qa-account-card"><AccountHeading account={account} /><Num value={account.equity} size="l" /><div className="qa-account-deltas"><Delta value={account.day_change} pct={account.day_change_pct} size="s" /><Delta value={account.week_change} pct={account.week_change_pct} size="s" /></div>{account.spark_30d && account.spark_30d.length > 1 && <Sparkline values={account.spark_30d} label="Valor últimos 30 días" tone={account.spark_30d.at(-1)! >= account.spark_30d[0] ? 'positive' : 'negative'} />}<footer>{account.alarms.length ? `${account.alarms.length} ${account.alarms.length === 1 ? 'alarma' : 'alarmas'}` : 'Sin alarmas'} · {time(account.as_of, 'ART')}</footer></Card>;
}

export function AccountSummaryCard({ account }: { account: AccountSummary }) {
  if (account.status === 'no_data') return <Card as="section" variant="dashed" className="qa-account-summary qa-account-no-data-card"><AccountHeading account={account} /><strong>Sin datos todavía</strong><p>{noDataReason(account.no_data_reason)}</p></Card>;
  return <Card as="section" className="qa-account-summary"><AccountHeading account={account} /><Num value={account.equity} size="xl" /><div className="qa-account-deltas"><Delta value={account.day_change} pct={account.day_change_pct} layout="stacked" /><Delta value={account.week_change} pct={account.week_change_pct} layout="stacked" /></div><footer className="qa-account-breakdown"><span>Largo <strong>{money(account.long_value)}</strong></span>{account.short_value > 0 && <span>Corto <strong>{money(account.short_value)}</strong></span>}<span>Efectivo <strong>{money(account.cash)}</strong></span><span>Órdenes de hoy <strong>{account.orders_today}</strong></span></footer></Card>;
}

function noDataReason(reason: AccountSummaryNoData['no_data_reason']) {
  if (reason === 'not_opened') return 'La cuenta todavía no se abrió en el broker.';
  if (reason === 'not_funded') return 'La cuenta existe pero todavía no tiene fondos.';
  return 'Está en la configuración, pero el broker todavía no informó nada: la cuenta no se abrió o no tiene fondos.';
}

export function AccountNoData({ account, otherAccounts = [], now = new Date() }: { account: AccountSummaryNoData; otherAccounts?: { id: string; name: string; href: string }[]; now?: Date }) {
  const next = account.next_check_at ? ` · Próxima: en ${Math.max(0, Math.round((new Date(account.next_check_at).getTime() - now.getTime()) / 60000))} min` : '';
  return <section className="qa-account-empty"><span className="qa-account-empty-icon" aria-hidden="true">◌</span><h2>La cuenta {account.name} todavía no tiene datos</h2><p>{noDataReason(account.no_data_reason)} No es un error.</p><p>Cuando llegue el primer dato vas a ver acá el valor, las posiciones y las órdenes.</p><div className="qa-account-check">Última consulta al broker: {time(account.checked_at, 'ART')}{next}</div>{otherAccounts.length > 0 && <div><strong>Mientras tanto:</strong><nav aria-label="Otras cuentas">{otherAccounts.map(other => <a key={other.id} href={other.href}>{other.name}</a>)}</nav></div>}</section>;
}
