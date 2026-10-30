import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Badge, Card, Delta, EmptyState, ErrorState, Notice, Num, Skeleton } from './index';

describe('primitivas', () => {
  it('Card admite link y variante', () => { render(<Card as="article" href="/x" variant="dashed">Cuenta</Card>); expect(screen.getByRole('link', { name: 'Cuenta' }).classList.contains('qa-card-dashed')).toBe(true); });
  it('Num y Delta formatean con es-AR y expresan dirección', () => { render(<><Num value={1240.5} /><Delta value={2} pct={1.2} /><Delta value={-2} /><Delta value={0} /></>); expect(screen.getByText('1.240,50')).toBeTruthy(); expect(screen.getByLabelText(/subió/)).toBeTruthy(); expect(screen.getByLabelText(/bajó/)).toBeTruthy(); expect(screen.getByLabelText(/sin cambio/)).toBeTruthy(); });
  it('Badge muestra texto e icono por estado', () => { render(<><Badge status="filled" /><Badge status="pending" /><Badge status="rejected" /><Badge status="cancelled" /><Badge status="expired" /></>); for (const text of ['Ejecutada', 'Pendiente', 'Rechazada', 'Cancelada', 'Vencida']) expect(screen.getByText(text)).toBeTruthy(); });
  it('Notice expone severidad accesible y texto', () => { render(<><Notice severity="critical" title="Error" message="API" /><Notice severity="warning" title="Atención" message="Aviso" /><Notice severity="stale" title="Antiguo" message="Dato" /></>); expect(screen.getByRole('alert')).toBeTruthy(); expect(screen.getByText('Atención')).toBeTruthy(); expect(screen.getByText('Antiguo')).toBeTruthy(); });
  it('Skeleton, vacío y error exponen sus estados', () => { render(<><Skeleton label="Cargando" /><EmptyState title="Vacío" message="Sin datos" /><ErrorState message="Falló" /></>); expect(screen.getByText('Cargando')).toBeTruthy(); expect(screen.getAllByRole('status').some(node => node.textContent?.includes('Vacío'))).toBe(true); expect(screen.getAllByRole('alert').some(node => node.textContent?.includes('Falló'))).toBe(true); });
});
