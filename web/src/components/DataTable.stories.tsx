import type { Story } from '@ladle/react';
import { DataTable } from './DataTable';
import '../styles/tokens.css';

const rows = [{ id: 'a', symbol: 'XLK', price: 210.5, change: 3.85, group: 'Rechazadas' }, { id: 'b', symbol: 'SPY', price: 602.2, change: -4.6, group: 'Resto' }];
const columns = [{ key: 'symbol', header: 'Símbolo', type: 'text' as const, sortable: true, card: 'title' as const }, { key: 'price', header: 'Precio USD', type: 'money' as const, sortable: true, defaultSortDir: 'desc' as const, card: 'primary' as const }, { key: 'change', header: 'Cambio %', type: 'pct' as const, sortable: true, card: 'secondary' as const }];
export const Ready: Story = () => <DataTable columns={columns} rows={rows} rowKey={row => row.id} caption="Mercado" state="ready" groupBy={row => ({ id: row.group, label: row.group, tone: row.group === 'Rechazadas' ? 'neg' : 'neutral' })} />;
Ready.meta = { title: 'Componentes / DataTable / Claro / Ready' };
export const ReadyDark: Story = () => <div data-theme="dark"><Ready /></div>;
ReadyDark.meta = { title: 'Componentes / DataTable / Oscuro / Ready' };
export const Loading: Story = () => <DataTable<typeof rows[number]> columns={columns} rows={[]} rowKey={row => row.id} caption="Mercado" state="loading" />;
export const Empty: Story = () => <DataTable<typeof rows[number]> columns={columns} rows={[]} rowKey={row => row.id} caption="Mercado" state="empty" />;
export const Error: Story = () => <DataTable<typeof rows[number]> columns={columns} rows={[]} rowKey={row => row.id} caption="Mercado" state="error" />;
export const Stale: Story = () => <DataTable columns={columns} rows={rows} rowKey={row => row.id} caption="Mercado" state="stale" staleAsOf="precios de las 13:05 ET" />;
export const StatesDark: Story = () => <div data-theme="dark"><div style={{ display: 'grid', gap: 'var(--qa-space-4)' }}><Loading /><Empty /><Error /><Stale /></div></div>;
