import { useState } from 'react';
import type { MacroSeries, MarketAsOf, MotorStateResponse, Regime, RegimeId, SectorRow } from '../api/types';
import { marketConfig, sectorHeatStep } from '../config';
import { dateLong, num, time } from '../lib/format';
import { Badge, Card, Delta, Notice } from './index';
import { DataTable, type Column } from './DataTable';
import './Market.css';

const labels: Record<RegimeId, string> = { up_calm: 'Alcista tranquilo', up_vol: 'Alcista agitado', down_calm: 'Bajista tranquilo', down_vol: 'Bajista agitado' };

export function MarketAsOfLine({ asOf }: { asOf: MarketAsOf }) {
  return <div className={`qa-market-as-of${asOf.stale ? ' is-stale' : ''}`}>
    <p>Al cierre del {dateLong(`${asOf.session_date}T12:00:00Z`)} · actualizado {time(asOf.updated_at, 'ART')}</p>
    {asOf.stale && <Notice severity="warning" title="Datos de mercado desactualizados" message={`Faltan ${asOf.missing_sessions.length} sesiones${asOf.missing_sessions.length ? `: ${asOf.missing_sessions.join(', ')}` : ''}.`} />}
  </div>;
}

function RegimeSegment({ regime, label, sessions }: { regime: RegimeId | null; label: string; sessions?: number }) {
  const text = regime ? labels[regime] : 'Sin datos';
  return <span className={`qa-regime-swatch ${regime ? `qa-reg-${regime.replace('_', '-')}` : 'qa-regime-missing'}`} role="img" aria-label={`${label}: ${text}${sessions === undefined ? '' : `, ${sessions} sesiones`}`} title={`${label}: ${text}`}>
    {sessions !== undefined && sessions >= 20 && (regime?.startsWith('up') ? '▲' : regime ? '▼' : '—')}
  </span>;
}

export function RegimeStrip({ strip }: { strip: Regime['strip'] }) {
  if (!strip.length) return <p className="qa-market-empty">Sin datos de régimen para los últimos 12 meses.</p>;
  return <div className="qa-regime-strip" aria-label="Régimen de SPY en los últimos 12 meses">
    <div className="qa-regime-segments">{strip.map((item, index) => <RegimeSegment key={`${item.from}-${index}`} regime={item.regime} label={`${dateLong(`${item.from}T12:00:00Z`)} a ${dateLong(`${item.to}T12:00:00Z`)}`} sessions={item.sessions} />)}</div>
    <div className="qa-regime-legend">{Object.entries(labels).map(([id, label]) => <span key={id}><RegimeSegment regime={id as RegimeId} label={label} />{label}</span>)}<span><RegimeSegment regime={null} label="Sin datos" />Sin datos</span></div>
    <div className="qa-regime-table" role="table" aria-label="Períodos de régimen">{strip.map((item, i) => <div role="row" key={`${item.from}-${i}`}><span role="cell">{dateLong(`${item.from}T12:00:00Z`)} – {dateLong(`${item.to}T12:00:00Z`)}</span><span role="cell">{item.regime ? labels[item.regime] : 'Sin datos'}</span><span role="cell">{item.sessions} sesiones</span></div>)}</div>
  </div>;
}

export function RegimeCard({ regime, stale = false }: { regime: Regime | null; stale?: boolean }) {
  if (!regime) return <Card as="section" className="qa-regime-card"><h2>Régimen de SPY</h2><p className="qa-market-empty">Sin datos de régimen: todavía no hay sesiones suficientes.</p></Card>;
  const up = regime.regime.startsWith('up');
  const highVol = regime.regime.endsWith('vol');
  return <Card as="section" className={`qa-regime-card${stale ? ' is-stale' : ''}`}>
    <h2>Régimen de SPY</h2><div className="qa-regime-summary"><RegimeSegment regime={regime.regime} label={labels[regime.regime]} /><div><strong>{labels[regime.regime]}</strong><p>Desde el {dateLong(`${regime.since}T12:00:00Z`)} · {regime.sessions} sesiones</p></div></div>
    <div className="qa-regime-facts"><div><strong>Tendencia</strong><span>{regime.ma200 == null ? 'Sin datos: hacen falta 200 sesiones para la media.' : `${up ? 'Alcista' : 'Bajista'} · precio ${up ? 'sobre' : 'bajo'} la media de 200 días`}</span></div><div><strong>Volatilidad</strong><span>{regime.vol20 == null || regime.vol_median_1y == null ? 'Sin datos: no se puede calcular el régimen.' : `${highVol ? 'Alta' : 'Baja'} · 20 días vs. mediana de 1 año`}</span></div></div>
    <RegimeStrip strip={regime.strip} />
  </Card>;
}

const motorLabels = { risk_on: 'Risk-on', neutral: 'Neutral', risk_off: 'Risk-off' } as const;

export function MotorStateCard({ motor, sectors }: { motor: MotorStateResponse; sectors: SectorRow[] }) {
  const enabled = sectors.filter(sector => sector.enabled).sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity)).slice(0, marketConfig.enabledTop);
  const blocked = sectors.filter(sector => sector.blocked);
  return <Card as="section" className="qa-motor-card">
    <h2>Motor M5</h2>
    {!motor ? <p className="qa-market-empty">El motor todavía no calculó.</p> : <>
      <div className="qa-motor-state"><strong>{motorLabels[motor.state]}</strong><span>Calculado el {time(motor.computed_at, 'ART')} · próximo cálculo {dateLong(`${motor.next_at.slice(0, 10)}T12:00:00Z`)}</span></div>
      {motor.state_description && <p className="qa-motor-description">{motor.state_description}</p>}
      <ol className="qa-motor-scale">{(['risk_off', 'neutral', 'risk_on'] as const).map(state => <li key={state} aria-current={motor.state === state ? 'step' : undefined}><strong>{motorLabels[state]}</strong><span>{state === 'risk_off' ? 'Sin compras nuevas' : state === 'neutral' ? 'Estrés macro elevado sin que SPY esté en risk-off' : 'El resto'}</span></li>)}</ol>
      <p>Sectores habilitados ({enabled.length}): {enabled.length ? enabled.map(s => s.symbol).join(', ') : 'ninguno'}</p>
      <p>Sectores bloqueados ({blocked.length}): {blocked.length ? blocked.map(s => `${s.symbol}${s.blocked_reason ? ` (${s.blocked_reason})` : ''}`).join(', ') : 'ninguno'}</p>
      <p className="qa-market-note">Es la regla del motor para las carteras paper, no una recomendación.</p>
    </>}
  </Card>;
}

export function MacroIndicatorCard({ series }: { series: MacroSeries }) {
  if (series.status === 'coming_soon') return <Card as="section" className="qa-macro-card qa-ghost"><header><h2>{series.title}</h2><Badge status="soon" /></header><div className="qa-macro-ghost" aria-label="Serie próximamente" /></Card>;
  const last = series.last;
  return <Card as="section" className={`qa-macro-card${series.stale ? ' is-stale' : ''}`}>
    <header><h2>{series.title}</h2><code>{series.code}</code></header>
    {series.stale && <Notice severity="warning" title="Dato desactualizado" message={`Se esperaba un dato el ${series.expected_release ?? 'según la frecuencia de la serie'}.`} />}
    {last ? <><strong className="qa-macro-value">{num(last.value)} {series.unit}</strong><Delta value={series.change_1m} missingLabel="Sin cambio mensual" /><p>Dato del {series.frequency === 'monthly' ? dateLong(`${last.observation_date}T12:00:00Z`) : dateLong(`${last.observation_date}T12:00:00Z`)}, publicado el {dateLong(`${last.published_at.slice(0, 10)}T12:00:00Z`)}</p></> : <p className="qa-market-empty">Sin datos publicados.</p>}
  </Card>;
}

const returnPeriods = [['d1', '1d'], ['w1', '1s'], ['m1', '1m'], ['m3', '3m'], ['ytd', 'YTD']] as const;
const periodLabels = { d1: '1 día', w1: '1 semana', m1: '1 mes', m3: '3 meses', ytd: 'año' } as const;
const heatLabels = ['≤ −3 umbrales', '−2 a −3 umbrales', '−1 a −2 umbrales', 'dentro del umbral neutro', '+1 a +2 umbrales', '+2 a +3 umbrales', '≥ +3 umbrales'];
const sectorColumns: Column<SectorRow>[] = [
  { key: 'sector', header: 'Sector', type: 'custom', sortable: true, sortValue: row => row.rank ?? Infinity, render: row => <span>{row.symbol} · {row.name}{row.is_reference && <small> · Referencia</small>}</span>, card: 'primary' },
  ...returnPeriods.map(([key, label]) => ({ key, header: label, type: 'custom' as const, sortable: true, sortValue: (row: SectorRow) => row.returns[key] ?? -Infinity, render: (row: SectorRow) => row.returns[key] == null ? <span title={`Sin datos para ${periodLabels[key]}`} aria-label={`Sin datos para ${periodLabels[key]}`}>—</span> : <span>{row.returns[key]! > 0 ? '+' : ''}{num(row.returns[key]!)}</span> })),
  { key: 'score', header: 'Score / ranking', type: 'custom', sortable: true, sortValue: row => row.rank ?? Infinity, render: row => row.is_reference ? 'Referencia' : row.momentum_score == null ? <span title="Score sin datos">Score: sin datos</span> : `${num(row.momentum_score)} · #${row.rank ?? '—'}` },
  { key: 'status', header: 'Estado', type: 'custom', render: row => row.is_reference ? 'Referencia' : row.blocked ? `Bloqueado${row.blocked_reason ? ` · ${row.blocked_reason}` : ''}` : row.enabled ? 'Habilitado' : 'Sin habilitar' },
  { key: 'held', header: 'En cartera', type: 'custom', render: row => row.held_in.length ? row.held_in.map(item => item.account_name).join(', ') : row.enabled ? 'Sin posición todavía' : '—' },
];

export function SectorTable({ sectors, state = 'ready' }: { sectors: SectorRow[]; state?: 'ready' | 'loading' | 'empty' | 'error' | 'stale' }) {
  const rows = [...sectors.filter(row => !row.is_reference).sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity)), ...sectors.filter(row => row.is_reference)];
  return <DataTable columns={sectorColumns} rows={rows} rowKey={row => row.symbol} caption="Retornos, estado y posiciones de sectores" state={state} />;
}

export function SectorHeatmap({ sectors }: { sectors: SectorRow[] }) {
  const [period, setPeriod] = useState<keyof typeof marketConfig.heatThresholds>('d1');
  const rows = [...sectors.filter(row => !row.is_reference).sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity)), ...sectors.filter(row => row.is_reference)];
  return <section className="qa-sector-heatmap">
    <label>Período <select aria-label="Período del mapa de calor" value={period} onChange={event => setPeriod(event.target.value as typeof period)}>{returnPeriods.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
    <div role="table" aria-label={`Mapa de calor sectorial, ${periodLabels[period]}`}><div role="row"><strong role="columnheader">Sector</strong><strong role="columnheader">Retorno {returnPeriods.find(([key]) => key === period)?.[1]}</strong><strong role="columnheader">Estado</strong></div>
      {rows.map(row => { const value = row.returns[period]; const step = value == null ? null : sectorHeatStep(value, period); return <div role="row" key={row.symbol}><span role="cell">{row.symbol} · {row.name}{row.is_reference ? ' · Referencia' : ''}</span><span role="cell" className={step == null ? '' : `qa-heat-step-${step}`} aria-label={value == null ? `Sin datos para ${periodLabels[period]}` : `${value > 0 ? '+' : ''}${num(value)} por ciento, ${heatLabels[step!]}`} title={value == null ? 'Sin datos' : heatLabels[step!]}>{value == null ? '—' : `${value > 0 ? '+' : ''}${num(value)}%`}</span><span role="cell">{row.is_reference ? 'Referencia' : row.blocked ? `Bloqueado${row.blocked_reason ? ` · ${row.blocked_reason}` : ''}` : row.enabled ? 'Habilitado' : 'Sin habilitar'}</span></div>; })}
    </div>
  </section>;
}
