import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { Regime } from '../api/types';
import { ThemeProvider } from '../ThemeContext';
import { MarketAsOfLine, RegimeCard, RegimeStrip } from './Market';

const themed = (ui: React.ReactNode) => render(<ThemeProvider>{ui}</ThemeProvider>);
const sample = (regime: Regime['regime']): Regime => ({ symbol: 'SPY', regime, since: '2026-10-01', sessions: 20, price: 100, ma200: 90, vol20: 0.2, vol_median_1y: 0.1, strip: [] });

describe('Market components', () => {
  afterEach(cleanup);
  it('shows the update time and dims stale market figures with a missing-session notice', () => {
    const { container } = themed(<><MarketAsOfLine asOf={{ session_date: '2026-11-02', updated_at: '2026-11-03T14:00:00Z', stale: true, missing_sessions: ['2026-11-03'] }} /><RegimeCard regime={sample('up_calm')} stale /></>);
    expect(screen.getByText(/actualizado .* ART/)).toBeTruthy();
    expect(screen.getByText(/Faltan 1 sesiones/)).toBeTruthy();
    expect(container.querySelector('.qa-regime-card')?.classList.contains('is-stale')).toBe(true);
  });
  it('renders an explicit empty state for a missing regime', () => {
    themed(<RegimeCard regime={null} />);
    expect(screen.getByText(/Sin datos de régimen/)).toBeTruthy();
  });
  it.each([['up_calm', 'Alcista tranquilo'], ['up_vol', 'Alcista agitado'], ['down_calm', 'Bajista tranquilo'], ['down_vol', 'Bajista agitado']] as const)('labels %s in text and accessible descriptions', (id, label) => {
    themed(<RegimeCard regime={sample(id)} />);
    expect(screen.getByText(label)).toBeTruthy();
    expect(screen.getByRole('img', { name: new RegExp(label) })).toBeTruthy();
  });
  it('marks unknown strip intervals as missing data', () => {
    themed(<RegimeStrip strip={[{ regime: null, from: '2026-11-01', to: '2026-11-02', sessions: 2 }]} />);
    expect(screen.getAllByText('Sin datos').length).toBeGreaterThan(0);
    expect(screen.getAllByRole('img', { name: /Sin datos/ }).length).toBeGreaterThan(0);
  });
});
