import type { Story } from '@ladle/react';
import { Badge, Card, ChipGroup, ComingSoon, Delta, EmptyState, ErrorState, HelpTip, Notice, Num, SegmentedControl, Skeleton } from './index';
import '../styles/tokens.css';

export const Primitives: Story = () => <div style={{ display: 'grid', gap: 'var(--qa-space-4)', maxWidth: 480 }}>
  <Card><Num value={1240.5} size="l" /></Card><Delta value={1240.5} pct={1.28} /><Delta value={-312.8} pct={-0.94} /><Delta value={0} pct={0} />
  {(['filled', 'pending', 'partial', 'rejected', 'cancelled', 'expired', 'paper', 'live', 'no-data', 'long', 'short', 'stage', 'candidate', 'production', 'discarded', 'soon'] as const).map(status => <Badge key={status} status={status} />)}
  <Notice severity="info" title="Información" message="Estado normal" /><Notice severity="stale" title="Desactualizado" message="Sin datos nuevos" since="14:05 ET" />
  <Skeleton label="Sigue cargando, la API está lenta" /><EmptyState title="Sin posiciones abiertas" message="Todavía no hay posiciones." /><ErrorState message="Revisá Tailscale y el servidor." action={{ label: 'Reintentar' }} />
  <HelpTip term="stop_distance" vars={{ symbol: 'XLK', pct: '2 %', amount: 'USD 12' }} />
  <ComingSoon title="Embudo" /><ComingSoon title="Mercado · ejemplo" example />
  <SegmentedControl label="Período" value="30" onChange={() => {}} options={[{ value: '7', label: '7 días' }, { value: '30', label: '30 días' }, { value: '90', label: '90 días' }, { value: 'all', label: 'Todo' }, { value: 'disabled', label: 'No disponible', disabled: true }]} />
  <ChipGroup label="Estado de órdenes" selected={['pending']} onChange={() => {}} options={[{ value: 'all', label: 'Todas', count: 8 }, { value: 'pending', label: 'Pendientes', count: 2 }, { value: 'empty', label: 'Rechazadas', count: 0 }]} />
</div>;
Primitives.meta = { title: 'Componentes / Primitivas' };
export const Dark: Story = () => <div data-theme="dark"><Primitives /></div>;
Dark.meta = { title: 'Componentes / Primitivas / Oscuro' };
export const ControlStates: Story = () => <div style={{ display: 'grid', gap: 'var(--qa-space-4)', maxWidth: 480 }}><SegmentedControl label="Estados" value="selected" onChange={() => {}} options={[{ value: 'normal', label: 'Normal' }, { value: 'selected', label: 'Seleccionado' }, { value: 'disabled', label: 'Deshabilitado', disabled: true }]} /><ChipGroup label="Estados de filtro" selected={['selected']} onChange={() => {}} options={[{ value: 'normal', label: 'Normal', count: 3 }, { value: 'selected', label: 'Seleccionado', count: 1 }, { value: 'disabled', label: 'Deshabilitado', count: 0 }]} /></div>;
ControlStates.meta = { title: 'Componentes / Controles / Estados' };
export const ControlStatesDark: Story = () => <div data-theme="dark"><ControlStates /></div>;
ControlStatesDark.meta = { title: 'Componentes / Controles / Estados / Oscuro' };
