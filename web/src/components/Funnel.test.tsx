import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { CycleMilestones, ReviewCounter, StageVerdictBadge } from './Funnel';

afterEach(cleanup);
describe('StageVerdictBadge', () => {
  it.each(['PASA', 'CORTA', 'ERROR', 'ALARMA'] as const)('expone %s como texto accesible', verdict => { render(<StageVerdictBadge verdict={verdict} />); expect(screen.getByLabelText(verdict).textContent).toContain(verdict); });
  it('marca la promoción con texto además del ícono', () => { render(<StageVerdictBadge verdict="PASA" promoted />); expect(screen.getByLabelText('PASA, promovida').textContent).toContain('PROMOVIDA'); });
});
describe('ReviewCounter', () => {
  it('maneja datos nulos y estados de carga y error', () => { const { rerender } = render(<ReviewCounter rule={null} />); expect(screen.getByText(/Sin datos/)).toBeTruthy(); rerender(<ReviewCounter rule={null} loading />); expect(screen.getByRole('status').textContent).toContain('Cargando'); rerender(<ReviewCounter rule={null} error />); expect(screen.getByRole('alert').textContent).toContain('No se pudo'); });
  it('usa el límite configurado y anuncia cuando se alcanza', () => { render(<ReviewCounter rule={{ limit: 2, streak: 6, reached: true, last_cycles: [{ cycle: '2026-09', promoted: 0 }, { cycle: '2026-10', promoted: 0 }] }} />); expect(screen.getByLabelText(/6 de 6.*alcanzada/)).toBeTruthy(); });
});
describe('CycleMilestones', () => {
  it('calcula hitos cuando el contrato trae null y los marca vencidos si corresponde', () => { render(<CycleMilestones milestones={null} cycle="2000-11" status="open" />); expect(screen.getByText('Propuesta')).toBeTruthy(); expect(screen.getByText('Arranque del embudo')).toBeTruthy(); expect(screen.getByText('Revisión de paper')).toBeTruthy(); expect(screen.getAllByText('vencido')).toHaveLength(3); });
  it('muestra como hechos los hitos recibidos', () => { render(<CycleMilestones milestones={[{ key: 'proposal', day: 1, date: '2026-11-01', done: true }]} cycle="2026-11" status="closed" />); expect(screen.getByText('hecho')).toBeTruthy(); });
});
