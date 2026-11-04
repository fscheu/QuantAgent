import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { MacroSeries, MotorState, Regime } from '../api/types';
import { ThemeProvider } from '../ThemeContext';
import { MarketAsOfLine, RegimeCard, RegimeStrip, MotorStateCard, MacroIndicatorCard } from './Market';

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
  it('shows no invented motor details before its first calculation', () => {
    themed(<MotorStateCard motor={null} sectors={[]} />);
    expect(screen.getByText('El motor todavía no calculó.')).toBeTruthy();
    expect(screen.queryByText(/Sectores habilitados/)).toBeNull();
  });
  it('lists the configured top enabled sectors and blocked sectors with text labels', () => {
    const sectors = ['XLK', 'XLF', 'XLY', 'XLE'].map((symbol, i) => ({ symbol, name: symbol, is_reference: false, held_in: [], returns: { d1: null, w1: null, m1: null, m3: null, ytd: null }, momentum_score: 1, rank: i + 1, enabled: i < 3, blocked: i === 3, blocked_reason: i === 3 ? 'score negativo' : null, score_as_of: null }));
    const motor: MotorState = { state: 'neutral', computed_at: '2026-11-04T14:00:00Z', next_at: '2026-11-11T14:00:00Z', state_description: null };
    themed(<MotorStateCard motor={motor} sectors={sectors} />);
    expect(screen.getByText(/Sectores habilitados \(3\): XLK, XLF, XLY/)).toBeTruthy();
    expect(screen.getByText(/XLE \(score negativo\)/)).toBeTruthy();
    expect(screen.getByText(/Estrés macro elevado/)).toBeTruthy();
  });
  const macro: MacroSeries = { code: 'T10Y2Y', title: 'Curva del Tesoro', frequency: 'daily', status: 'active', unit: 'pp', last: { value: 0.12, observation_date: '2026-11-03', published_at: '2026-11-03T14:00:00Z' }, change_1m: 0.04, stale: false, expected_release: null, points: [] };
  it.each([false, true])('renders active macro values and stale state=%s with text', stale => {
    const { container } = themed(<MacroIndicatorCard series={{ ...macro, stale, expected_release: stale ? '2026-11-04' : null }} />);
    expect(screen.getByText(/Curva del Tesoro/)).toBeTruthy();
    expect(screen.getByText(/0.12 pp/)).toBeTruthy();
    expect(container.querySelector('.qa-macro-card')?.classList.contains('is-stale')).toBe(stale);
    if (stale) expect(screen.getByText('Dato desactualizado')).toBeTruthy();
  });
  it('renders coming soon macro as a value-free ghost', () => {
    const { container } = themed(<MacroIndicatorCard series={{ ...macro, status: 'coming_soon', last: null }} />);
    expect(screen.getByText('Próximamente')).toBeTruthy();
    expect(container.querySelector('.qa-ghost')).toBeTruthy();
    expect(screen.queryByText(/0.12/)).toBeNull();
  });
});
