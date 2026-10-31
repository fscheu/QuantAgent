import type { Order } from '../api/types';
import { money, num, time } from '../lib/format';
import { Badge, Card, HelpTip } from './index';
import type { Column } from './DataTable';
import './Order.css';

export function RejectReason({ order }: { order: Order }) {
  if (!order.reject_reason) return <div className="qa-reject-missing"><span aria-hidden="true">◌</span><strong>Motivo no disponible</strong><HelpTip term="reject_reason_missing" /></div>;
  return <div className="qa-reject-reason"><strong>{order.reject_reason}</strong>{order.reject_detail && <span>{order.reject_detail}</span>}{order.reject_reason_raw && <small>Broker: {order.reject_reason_raw}</small>}</div>;
}

function expiredExplanation(order: Order) {
  if (order.status !== 'expired') return null;
  return order.time_in_force === 'day'
    ? 'Orden del día: venció al cierre (16:00 ET) sin que el precio llegara al límite.'
    : `Venció el ${order.expired_at ? time(order.expired_at, 'ET') : 'sin fecha informada'} sin ejecutarse.`;
}

export function OrderCard({ order }: { order: Order }) {
  const explanation = expiredExplanation(order);
  return <Card as="article" className="qa-order-card">
    <header className="qa-order-heading"><div><strong>{order.symbol}</strong><span>{order.side === 'buy' ? 'Compra' : 'Venta'} · {num(order.qty)} · {time(order.submitted_at, 'ET')}</span></div><Badge status={order.status === 'partially_filled' ? 'partial' : order.status} /></header>
    <p>{order.type === 'market' ? 'Mercado' : order.type === 'limit' ? `Límite ${order.limit_price == null ? '—' : money(order.limit_price)}` : order.type === 'stop' ? `Stop ${order.stop_price == null ? '—' : money(order.stop_price)}` : 'Stop límite'}</p>
    {order.status === 'rejected' && <RejectReason order={order} />}
    {explanation && <p className="qa-order-expired">{explanation}</p>}
  </Card>;
}

export const orderColumns: Column<Order>[] = [
  { key: 'symbol', header: 'Símbolo', type: 'custom', width: '100px', card: 'title', render: order => <strong>{order.symbol}</strong> },
  { key: 'side', header: 'Lado', type: 'custom', width: '80px', render: order => order.side === 'buy' ? 'Compra' : 'Venta' },
  { key: 'qty', header: 'Cant.', type: 'num', width: '80px' },
  { key: 'type', header: 'Tipo / precio', type: 'custom', width: '1fr', render: order => order.type === 'market' ? 'Mercado' : order.type === 'limit' ? `Límite ${order.limit_price == null ? '—' : money(order.limit_price)}` : order.type === 'stop' ? `Stop ${order.stop_price == null ? '—' : money(order.stop_price)}` : 'Stop límite' },
  { key: 'status', header: 'Estado', type: 'custom', width: '120px', render: order => <Badge status={order.status === 'partially_filled' ? 'partial' : order.status} /> },
  { key: 'submitted_at', header: 'Enviada', type: 'custom', width: '130px', render: order => time(order.submitted_at, 'ET') },
];

export const OrderRow = orderColumns;
export const orderGroupBy = (order: Order) => order.status === 'rejected' ? { id: 'rejected', label: 'Rechazadas', tone: 'neg' as const } : { id: 'rest', label: 'Resto', tone: 'neutral' as const };
export const orderSubRow = (order: Order) => order.status === 'rejected' ? <RejectReason order={order} /> : expiredExplanation(order) ? <p className="qa-order-expired">{expiredExplanation(order)}</p> : null;
