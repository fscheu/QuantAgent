import type { Regime } from '../api/types';
import { MarketAsOfLine, RegimeCard } from './Market';
import { ThemeProvider } from '../ThemeContext';

const regime: Regime = { symbol: 'SPY', regime: 'up_vol', since: '2026-10-12', sessions: 16, price: 680, ma200: 620, vol20: 0.21, vol_median_1y: 0.16, strip: [{ regime: 'up_calm', from: '2026-06-01', to: '2026-08-31', sessions: 63 }, { regime: 'up_vol', from: '2026-09-01', to: '2026-11-03', sessions: 43 }, { regime: null, from: '2026-11-04', to: '2026-11-04', sessions: 0 }] };

export const DatosDeMercado = () => <ThemeProvider><MarketAsOfLine asOf={{ session_date: '2026-11-03', updated_at: '2026-11-04T01:00:00Z', stale: false, missing_sessions: [] }} /><RegimeCard regime={regime} /></ThemeProvider>;
export const SinRegimen = () => <ThemeProvider><RegimeCard regime={null} /></ThemeProvider>;
