import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { Position } from '../api/types';
import { PositionCard } from './Position';
import { StopDistanceBar } from './StopDistanceBar';

const props = { side: 'long' as const, price: 100, stop: 90, qty: 10 };

describe('StopDistanceBar', () => {
  it('calculates margin and reports meter value for a long position', () => {
    const { container } = render(<StopDistanceBar {...props} />);
    const meter = screen.getByRole('meter');
    expect(meter.getAttribute('aria-valuenow')).toBe('10');
    expect(meter.getAttribute('aria-valuetext')).toBe('a 10,00 por ciento del stop, stop por debajo del precio');
    expect((container.querySelector('.qa-stop-fill') as HTMLElement).style.width).toBe('66.66666666666666%');
    expect(screen.getByText('Margen · 10,00 % · USD 100,00')).toBeTruthy();
  });
  it('shows near, attention, breached and no-stop states', () => {
    const { container, rerender, unmount } = render(<StopDistanceBar {...props} stop={98.5} />);
    expect(screen.getByText('Cerca · 1,50 % · USD 15,00')).toBeTruthy();
    rerender(<StopDistanceBar {...props} stop={96} />);
    expect(screen.getByText('Atención · 4,00 % · USD 40,00')).toBeTruthy();
    expect((container.querySelector('.qa-stop-fill') as HTMLElement).style.width).toBe(`${(4 / 15) * 100}%`);
    expect((container.querySelector('[role="meter"]') as HTMLElement).getAttribute('aria-valuetext')).toBe('a 4,00 por ciento del stop, stop por debajo del precio');
    unmount();
    const shortView = render(<StopDistanceBar side="short" price={100} stop={101.5} qty={10} />);
    const shortMeter = shortView.container.querySelector('[role="meter"]') as HTMLElement;
    expect(shortMeter.getAttribute('aria-valuetext')).toBe('a 1,50 por ciento del stop, stop por encima del precio');
    expect(shortMeter.classList.contains('qa-stop-anchor-right')).toBe(true);
    expect(screen.getByText('Cerca · 1,50 % · USD 15,00')).toBeTruthy();
    shortView.rerender(<StopDistanceBar {...props} stop={102} />);
    expect(screen.getByText('Stop superado · USD 20,00')).toBeTruthy();
    expect(shortView.container.querySelector('.qa-stop')?.classList.contains('qa-stop-breached')).toBe(true);
    expect((shortView.container.querySelector('[role="meter"]') as HTMLElement).getAttribute('aria-valuetext')).toBe('precio por debajo del stop');
    shortView.rerender(<StopDistanceBar {...props} stop={null} />);
    expect(screen.getAllByText('Sin stop vigente').length).toBeGreaterThan(0);
    expect((shortView.container.querySelector('[role="meter"]') as HTMLElement).getAttribute('aria-valuetext')).toBe('Sin stop vigente');
  });
});

describe('PositionCard', () => {
  const shortPosition: Position = { symbol: 'XYZ', description: 'Empresa XYZ', side: 'short', qty: 5, avg_entry_price: 110, current_price: 100, market_value: 500, unrealized_pl: 50, unrealized_pl_pct: 9.09, stop_price: 120, opened_at: '2026-10-20' };
  it('explains a short profit when price falls', () => {
    render(<PositionCard position={shortPosition} />);
    expect(screen.getByText('Gana si el precio baja: bajó de 110,00 a 100,00.')).toBeTruthy();
  });
  it('explains a short loss when price rises', () => {
    render(<PositionCard position={{ ...shortPosition, current_price: 120, unrealized_pl: -50 }} />);
    expect(screen.getByText('Pierde si el precio sube: subió de 110,00 a 120,00.')).toBeTruthy();
  });
  it('shows no current stop for a long position without a stop', () => {
    const { container } = render(<PositionCard position={{ ...shortPosition, side: 'long', qty: 10, stop_price: null }} />);
    expect(container.querySelector('.qa-stop-copy')?.textContent).toContain('Sin stop vigente');
  });
  it('shows absolute value and positive quantity for a negative short quantity', () => {
    const { container } = render(<PositionCard position={{ ...shortPosition, qty: -5, market_value: -500 }} />);
    expect(container.querySelector('.qa-position-value')?.textContent).toContain('500,00');
    expect(container.querySelector('.qa-position-value')?.textContent).toContain('5,00 × USD 100,00');
  });
});
