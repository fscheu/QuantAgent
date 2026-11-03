import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { AccountSummary, Order } from '../api/types';
import { AccountCard, AccountNoData, AccountSummaryCard } from './Account';
import { DataTable } from './DataTable';
import { OrderCard, orderColumns, orderGroupBy, orderSubRow, RejectReason } from './Order';

const active: Extract<AccountSummary, { status: 'active' }> = { id: 'a', name: 'Paper', env: 'paper', status: 'active', as_of: '2026-10-30T12:00:00Z', stale: false, equity: 1000, long_value: 800, short_value: 0, cash: 200, day_change: 1, day_change_pct: 0.001, week_change: 2, week_change_pct: 0.002, positions_count: 1, orders_today: 2, orders_rejected_today: 0, alarms: [], spark_30d: null };
const noData: Extract<AccountSummary, { status: 'no_data' }> = { id: 'b', name: 'Live', env: 'live', status: 'no_data', no_data_reason: 'not_funded', checked_at: '2026-10-30T12:00:00Z', next_check_at: null };
const order: Order = { id: '1', symbol: 'AAPL', side: 'buy', type: 'limit', qty: 1, limit_price: 100, stop_price: null, status: 'rejected', filled_qty: 0, filled_avg_price: null, time_in_force: 'day', submitted_at: '2026-10-30T12:00:00Z', cancelled_at: null, expired_at: null, reject_reason: 'Fondos insuficientes', reject_detail: 'Faltaban fondos.', reject_reason_raw: 'raw reason' };

describe('Account cards', () => {
  it.each([AccountCard, AccountSummaryCard])('formats fractions and distinguishes missing history from zero', (Card) => {
    const { container, rerender, unmount } = render(<Card account={{ ...active, day_change_pct: 0.00029, week_change: null, week_change_pct: null }} />);
    expect(screen.getByText('+0,03 %')).toBeTruthy();
    expect(screen.getByLabelText('subió 1,00 dólares, 0,03 por ciento')).toBeTruthy();
    const missing = screen.getByText('Sin 7 días de historial').closest('.qa-delta')!;
    expect(missing.textContent).toBe('—Sin 7 días de historial');
    expect(missing.classList.contains('qa-neutral')).toBe(false);
    rerender(<Card account={{ ...active, day_change: null, day_change_pct: null, week_change: 0, week_change_pct: 0 }} />);
    expect(screen.getByText('Sin historial diario')).toBeTruthy();
    expect(screen.getByText('0,00 %')).toBeTruthy();
    expect(screen.getByLabelText('sin cambio 0,00 dólares, 0,00 por ciento')).toBeTruthy();
    rerender(<Card account={{ ...active, day_change_pct: null, week_change_pct: null }} />);
    expect(container.querySelector('.qa-account-deltas')?.textContent).not.toMatch(/%|NaN/);
    expect(screen.getByLabelText('subió 1,00 dólares')).toBeTruthy();
    unmount();
  });
  it('never renders amounts or zeros for no data', () => {
    const { container, rerender } = render(<AccountCard account={noData} />);
    expect(container.textContent).not.toMatch(/0,00|USD\s*0/);
    rerender(<AccountNoData account={noData} />);
    expect(container.textContent).not.toMatch(/0,00|USD\s*0/);
    expect(screen.queryByText(/Próxima/)).toBeNull();
  });
  it('uses one no-data badge and calculates the next check from its timestamp', () => {
    const nextAt = (next_check_at: string | null) => ({ ...noData, next_check_at });
    const now = new Date('2026-10-30T12:00:00Z');
    const { container, rerender } = render(<AccountCard account={noData} />);
    expect(container.querySelectorAll('.qa-badge')).toHaveLength(1);
    expect(container.querySelector('.qa-badge-no-data')).toBeTruthy();
    rerender(<AccountSummaryCard account={noData} />);
    expect(container.querySelectorAll('.qa-badge')).toHaveLength(1);
    expect(container.querySelector('.qa-badge-no-data')).toBeTruthy();
    rerender(<AccountNoData account={nextAt('2026-10-30T12:15:00Z')} now={now} />);
    expect(screen.getByText(/Próxima: en 15 min/)).toBeTruthy();
    rerender(<AccountNoData account={nextAt('2026-10-30T12:07:00Z')} now={now} />);
    expect(screen.getByText(/Próxima: en 7 min/)).toBeTruthy();
    rerender(<AccountNoData account={nextAt(null)} now={now} />);
    expect(screen.queryByText(/Próxima/)).toBeNull();
  });
  it('renders the active account time without a currency symbol', () => {
    const { container } = render(<AccountCard account={active} />);
    expect(container.querySelector('footer')?.textContent).toContain('Sin alarmas ·');
    expect(container.querySelector('footer')?.textContent).not.toContain('$');
  });
  it('omits short value when there are no short positions', () => {
    render(<AccountSummaryCard account={active} />);
    expect(screen.queryByText('Corto')).toBeNull();
  });
});

describe('orders', () => {
  it('shows broker detail and help tip for present and missing reject reasons', () => {
    const { rerender } = render(<RejectReason order={order} />);
    expect(screen.getByText('Fondos insuficientes')).toBeTruthy();
    expect(screen.getByText('Broker: raw reason')).toBeTruthy();
    rerender(<RejectReason order={{ ...order, reject_reason: null, reject_detail: null, reject_reason_raw: null }} />);
    expect(screen.getByText('Motivo no disponible')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Qué es Motivo no disponible' })).toBeTruthy();
  });
  it('explains day and other expired orders and groups rejected rows first', () => {
    const { rerender } = render(<OrderCard order={{ ...order, status: 'expired', reject_reason: null, reject_detail: null, reject_reason_raw: null }} />);
    expect(screen.getByText(/venció al cierre \(16:00 ET\)/)).toBeTruthy();
    rerender(<OrderCard order={{ ...order, status: 'expired', time_in_force: 'gtc', expired_at: '2026-10-30T12:00:00Z', reject_reason: null, reject_detail: null, reject_reason_raw: null }} />);
    expect(screen.getByText(/Venció el/)).toBeTruthy();
    render(<DataTable columns={orderColumns} rows={[{ ...order, status: 'pending' }, order]} rowKey={row => row.id} caption="Órdenes" state="ready" groupBy={orderGroupBy} subRow={orderSubRow} />);
    expect(screen.getByRole('heading', { name: 'Rechazadas · 1' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Resto · 1' })).toBeTruthy();
  });
});
