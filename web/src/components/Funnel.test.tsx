import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { CycleMilestones, DropReasons, FunnelChart, ReviewCounter, StageVerdictBadge } from './Funnel';
import type { Funnel } from '../api/types';

afterEach(cleanup);
const funnel: Funnel = { cycle: 'all', period: 'all', total: 4, sample_too_small: false, promoted: 1, stages: (['e0', 'e1', 'e2', 'e3'] as const).map((stage, i) => ({ stage, reached: 4 - i, here: 1, passed: 2, cut: 1, error: i === 1 ? 1 : 0, alarm: i === 2 ? 1 : 0 })), later: [{ phase: 'shadow', reached: 2, here: 1 }, { phase: 'paper', reached: 1, here: 1 }, { phase: 'candidate', reached: 0, here: 0 }], drop_reasons: [] };
describe('FunnelChart', () => {
  it('muestra etapas, fases y segmentos de detención con conteos accesibles', () => { render(<FunnelChart funnel={funnel} />); expect(screen.getByRole('img', { name: /E1: PASA 2, CORTA 1, ERROR 1/ })).toBeTruthy(); expect(screen.getByRole('img', { name: /E2: PASA 2, CORTA 1, ERROR 0, ALARMA 1/ })).toBeTruthy(); expect(screen.getByText(/Sombra local: 2/)).toBeTruthy(); expect(screen.getByText(/1 promovidas/)).toBeTruthy(); });
  it('explica cuando no hay muestra suficiente', () => { const { rerender } = render(<FunnelChart funnel={{ ...funnel, total: 1 }} />); expect(screen.getByRole('status').textContent).toContain('Muestra chica'); rerender(<FunnelChart funnel={{ ...funnel, sample_too_small: true }} />); expect(screen.getByRole('status')).toBeTruthy(); });
});
describe('DropReasons', () => {
  it('muestra causa, veredictos, medida y omite el ejemplo nulo', () => { render(<DropReasons reasons={[{ stage: 'e1', verdict: 'CORTA', cause_code: 'sharpe', cause_label: 'Sharpe bajo el umbral', count: 2, example: { metric: 'Sharpe', value: 0.31, threshold: 0.5, unit: 'ratio', comparator: '>=' } }, { stage: 'e2', verdict: 'ERROR', cause_code: 'load', cause_label: 'Error de carga', count: 1, example: null }, { stage: 'e3', verdict: 'ALARMA', cause_code: 'control', cause_label: 'Control', count: 1, example: null }]} />); expect(screen.getByText('Sharpe bajo el umbral')).toBeTruthy(); expect(screen.getByText(/0,31.*umbral >= 0,5/)).toBeTruthy(); expect(screen.getByLabelText('ERROR')).toBeTruthy(); expect(screen.getByLabelText('ALARMA')).toBeTruthy(); expect(screen.getAllByText(/Error de carga|Control/)).toHaveLength(2); });
  it('explicita la lista vacía', () => { render(<DropReasons reasons={[]} />); expect(screen.getByRole('status').textContent).toContain('Ninguna cayó'); });
});
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
