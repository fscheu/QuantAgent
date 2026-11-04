import type { MacroSeries, MotorState, Regime, SectorRow } from '../api/types';
import { MarketAsOfLine, RegimeCard, MotorStateCard, MacroIndicatorCard } from './Market';
import { ThemeProvider } from '../ThemeContext';

const regime: Regime = { symbol: 'SPY', regime: 'up_vol', since: '2026-10-12', sessions: 16, price: 680, ma200: 620, vol20: 0.21, vol_median_1y: 0.16, strip: [{ regime: 'up_calm', from: '2026-06-01', to: '2026-08-31', sessions: 63 }, { regime: 'up_vol', from: '2026-09-01', to: '2026-11-03', sessions: 43 }, { regime: null, from: '2026-11-04', to: '2026-11-04', sessions: 0 }] };

export const DatosDeMercado = () => <ThemeProvider><MarketAsOfLine asOf={{ session_date: '2026-11-03', updated_at: '2026-11-04T01:00:00Z', stale: false, missing_sessions: [] }} /><RegimeCard regime={regime} /></ThemeProvider>;
export const SinRegimen = () => <ThemeProvider><RegimeCard regime={null} /></ThemeProvider>;
const motor: MotorState = { state: 'risk_on', computed_at: '2026-11-04T14:00:00Z', next_at: '2026-11-11T14:00:00Z', state_description: null };
const sectors = [{ symbol: 'XLK', enabled: true, rank: 1, blocked: false, blocked_reason: null }, { symbol: 'XLE', enabled: false, rank: null, blocked: true, blocked_reason: 'score negativo' }] as SectorRow[];
const macro: MacroSeries = { code: 'T10Y2Y', title: 'Curva del Tesoro', frequency: 'daily', status: 'active', unit: 'pp', last: { value: 0.12, observation_date: '2026-11-03', published_at: '2026-11-03T14:00:00Z' }, change_1m: 0.04, stale: false, expected_release: null, points: [] };
export const EstadoMotor = () => <ThemeProvider><MotorStateCard motor={motor} sectors={sectors} /></ThemeProvider>;
export const IndicadoresMacro = () => <ThemeProvider><MacroIndicatorCard series={macro} /><MacroIndicatorCard series={{ ...macro, code: 'CPIAUCSL', title: 'Inflación', status: 'coming_soon', last: null }} /></ThemeProvider>;
