import type { MarketAsOf, Regime, RegimeId } from '../api/types';
import { dateLong, time } from '../lib/format';
import { Card, Notice } from './index';
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
