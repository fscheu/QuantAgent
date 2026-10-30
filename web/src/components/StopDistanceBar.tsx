import { money, num } from '../lib/format';
import { stopConfig, stopDistance, type PositionSide } from '../config';
import './Position.css';

export interface StopDistanceBarProps {
  side: PositionSide;
  price: number;
  stop: number | null;
  qty: number;
  variant?: 'compact' | 'anatomy';
}

const zoneText = { near: 'Cerca', attention: 'Atención', ok: 'Margen', breached: 'Stop superado', none: 'Sin stop vigente' } as const;
const zoneIcon = { near: '⯃', attention: '△', ok: '○', breached: '⯃', none: '◌' } as const;

export function StopDistanceBar({ side, price, stop, qty, variant = 'compact' }: StopDistanceBarProps) {
  const distance = stopDistance(side, price, stop, qty);
  const pct = distance.pct === null ? null : Math.abs(distance.pct);
  const position = side === 'short' ? 'por encima' : 'por debajo';
  const valueText = distance.zone === 'none'
    ? 'Sin stop vigente'
    : distance.zone === 'breached'
      ? `precio ${side === 'long' ? 'por debajo' : 'por encima'} del stop`
      : `a ${num(pct!)} por ciento del stop, stop ${position} del precio`;
  const label = distance.zone === 'none'
    ? zoneText.none
    : distance.zone === 'breached'
      ? `${zoneText.breached} · ${money(distance.amount!)}`
      : `${zoneText[distance.zone]} · ${num(pct!)} % · ${money(distance.amount!)}`;
  return <div className={`qa-stop qa-stop-${variant} qa-stop-${distance.zone}`}>
    <div className="qa-stop-copy"><span className="qa-stop-icon" aria-hidden="true">{zoneIcon[distance.zone]}</span><span>{label}</span></div>
    <div className={`qa-stop-track qa-stop-anchor-${distance.anchor}`} role="meter" aria-label="Distancia al stop" aria-valuemin={0} aria-valuemax={stopConfig.scalePct} aria-valuenow={pct ?? 0} aria-valuetext={valueText}>
      <span className="qa-stop-fill" style={{ width: `${distance.fill * 100}%` }} />
    </div>
    {variant === 'anatomy' && stop !== null && <div className="qa-stop-anatomy">Stop {side === 'long' ? 'abajo' : 'arriba'} · {num(stop)}</div>}
  </div>;
}
