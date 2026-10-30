import type { Story } from '@ladle/react';
import type { AccountSummary, Order } from '../api/types';
import { AccountCard, AccountNoData, AccountSummaryCard } from './Account';
import { DataTable } from './DataTable';
import { OrderCard, orderColumns, orderGroupBy, orderSubRow } from './Order';
import '../styles/tokens.css';

const active: AccountSummary = { id: 'paper', name: 'Paper M4', env: 'paper', status: 'active', as_of: '2026-10-30T15:00:00Z', stale: false, equity: 12500, long_value: 8000, short_value: 0, cash: 4500, day_change: 120, day_change_pct: 0.96, week_change: -80, week_change_pct: -0.64, positions_count: 2, orders_today: 4, orders_rejected_today: 1, alarms: [], spark_30d: null };
const noData: AccountSummary = { id: 'live', name: 'Live', env: 'live', status: 'no_data', no_data_reason: 'not_opened', checked_at: '2026-10-30T15:00:00Z', next_check_at: '2026-10-30T15:15:00Z' };
const base: Order = { id: '1', symbol: 'AAPL', side: 'buy', type: 'limit', qty: 10, limit_price: 100, stop_price: null, status: 'rejected', filled_qty: 0, filled_avg_price: null, time_in_force: 'day', submitted_at: '2026-10-30T14:00:00Z', cancelled_at: null, expired_at: null, reject_reason: 'Fondos insuficientes', reject_detail: 'Necesitaba USD 1.000; había USD 500.', reject_reason_raw: 'insufficient buying power' };
const orders: Order[] = [base, { ...base, id: '2', symbol: 'MSFT', status: 'expired', reject_reason: null, reject_detail: null, reject_reason_raw: null, time_in_force: 'gtc', expired_at: '2026-10-29T20:00:00Z' }];

export const AccountStates: Story = () => <div style={{ display: 'grid', gap: 'var(--qa-space-4)' }}><AccountCard account={active} /><AccountCard account={noData} /><AccountSummaryCard account={active as Extract<AccountSummary, { status: 'active' }>} /><AccountNoData account={noData as Extract<AccountSummary, { status: 'no_data' }>} otherAccounts={[{ id: 'paper', name: 'Paper M4', href: '/cuenta/paper' }]} /></div>;
AccountStates.meta = { title: 'Componentes / Cuenta / Claro' };
export const AccountStatesDark: Story = () => <div data-theme="dark"><AccountStates /></div>;
AccountStatesDark.meta = { title: 'Componentes / Cuenta / Oscuro' };
export const OrderStates: Story = () => <div style={{ display: 'grid', gap: 'var(--qa-space-4)' }}><OrderCard order={base} /><DataTable columns={orderColumns} rows={orders} rowKey={order => order.id} caption="Órdenes" state="ready" groupBy={orderGroupBy} subRow={orderSubRow} /></div>;
OrderStates.meta = { title: 'Componentes / Órdenes / Claro' };
export const OrderStatesDark: Story = () => <div data-theme="dark"><OrderStates /></div>;
OrderStatesDark.meta = { title: 'Componentes / Órdenes / Oscuro' };
