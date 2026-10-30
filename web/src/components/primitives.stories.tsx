import type { Story } from '@ladle/react';
import { Badge, Card, Delta, EmptyState, ErrorState, Notice, Num, Skeleton } from './index';
import '../styles/tokens.css';

export const Primitives: Story = () => <div style={{ display: 'grid', gap: 'var(--qa-space-4)', maxWidth: 480 }}>
  <Card><Num value={1240.5} size="l" /></Card><Delta value={1240.5} pct={1.28} /><Delta value={-312.8} pct={-0.94} /><Delta value={0} pct={0} />
  {(['filled', 'pending', 'partial', 'rejected', 'cancelled', 'expired', 'paper', 'live', 'no-data', 'long', 'short', 'stage', 'candidate', 'production', 'discarded', 'soon'] as const).map(status => <Badge key={status} status={status} />)}
  <Notice severity="info" title="Información" message="Estado normal" /><Notice severity="stale" title="Desactualizado" message="Sin datos nuevos" since="14:05 ET" />
  <Skeleton label="Sigue cargando, la API está lenta" /><EmptyState title="Sin posiciones abiertas" message="Todavía no hay posiciones." /><ErrorState message="Revisá Tailscale y el servidor." action={{ label: 'Reintentar' }} />
</div>;
Primitives.meta = { title: 'Componentes / Primitivas' };
export const Dark: Story = () => <div data-theme="dark"><Primitives /></div>;
Dark.meta = { title: 'Componentes / Primitivas / Oscuro' };
