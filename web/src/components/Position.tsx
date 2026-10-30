import type { Position } from '../api/types';
import { money, num } from '../lib/format';
import { Card, Badge, Delta, Num } from './index';
import { StopDistanceBar } from './StopDistanceBar';
import type { Column } from './DataTable';
import './Position.css';

export function PositionCard({ position }: { position: Position }) {
  const short = position.side === 'short';
  const priceMoved = short ? position.current_price < position.avg_entry_price : position.current_price > position.avg_entry_price;
  const movement = short ? (priceMoved ? 'Gana si el precio baja' : 'Pierde si el precio sube') : '';
  const explanation = short ? `${movement}: ${priceMoved ? 'bajó' : 'subió'} de ${num(position.avg_entry_price)} a ${num(position.current_price)}.` : null;
  return <Card as="article" className="qa-position-card">
    <header className="qa-position-heading"><div><h3>{position.symbol}</h3>{position.description && <p>{position.description}</p>}</div><Badge status={position.side} /></header>
    <div className="qa-position-value"><Num value={Math.abs(position.market_value)} size="l" /><span>{num(Math.abs(position.qty))} × {money(position.current_price)} · entr. {money(position.avg_entry_price)}</span></div>
    <Delta value={position.unrealized_pl} pct={position.unrealized_pl_pct} />
    {explanation && <p className="qa-position-explanation">{explanation}</p>}
    <StopDistanceBar side={position.side} price={position.current_price} stop={position.stop_price} qty={Math.abs(position.qty)} variant="compact" />
  </Card>;
}

const distanceSort = (position: Position) => position.stop_price == null ? Number.POSITIVE_INFINITY : Math.abs((position.side === 'short' ? position.stop_price - position.current_price : position.current_price - position.stop_price) / position.current_price);

export const positionColumns: Column<Position>[] = [
  { key: 'symbol', header: 'Símbolo', type: 'custom', width: '120px', card: 'title', render: p => <span className="qa-position-symbol"><strong>{p.symbol}</strong>{p.description && <small>{p.description}</small>}</span> },
  { key: 'side', header: 'Lado', type: 'custom', width: '80px', card: 'secondary', render: p => <Badge status={p.side} /> },
  { key: 'qty', header: 'Cant.', type: 'custom', width: '44px', card: 'secondary', render: p => <Num value={Math.abs(p.qty)} size="s" /> },
  { key: 'avg_entry_price', header: 'Entrada', type: 'custom', width: '76px', card: 'secondary', render: p => <span className="qa-num">{money(p.avg_entry_price)}</span> },
  { key: 'current_price', header: 'Actual', type: 'custom', width: '76px', card: 'secondary', render: p => <span className="qa-num">{money(p.current_price)}</span> },
  { key: 'market_value', header: 'Valor', type: 'custom', width: '96px', card: 'primary', render: p => <span className="qa-num">{money(Math.abs(p.market_value))}</span> },
  { key: 'unrealized_pl', header: 'P&L no realizado', type: 'custom', width: '156px', card: 'secondary', render: p => <Delta value={p.unrealized_pl} pct={p.unrealized_pl_pct} size="s" /> },
  { key: 'stop_price', header: 'Stop', type: 'custom', width: '68px', card: 'secondary', render: p => p.stop_price === null ? '—' : <span className="qa-num">{money(p.stop_price)}</span> },
  { key: 'stop_distance', header: 'Distancia al stop', type: 'custom', width: 'minmax(190px,1fr)', card: 'secondary', sortable: true, defaultSortDir: 'asc', sortValue: distanceSort, render: p => <StopDistanceBar side={p.side} price={p.current_price} stop={p.stop_price} qty={Math.abs(p.qty)} variant="compact" /> },
  { key: 'opened_at', header: 'Apertura', type: 'date', width: '84px', card: 'secondary' },
];

export const PositionRow = positionColumns;
