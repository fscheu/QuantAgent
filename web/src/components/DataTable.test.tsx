import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DataTable, type Column } from './DataTable';

const data = [{ id: 'a', name: 'Alfa', value: 3 }, { id: 'b', name: 'Beta', value: 1 }, { id: 'c', name: 'Gamma', value: 2 }];
const columns: Column<typeof data[number]>[] = [{ key: 'name', header: 'Nombre', type: 'text', sortable: true, card: 'title' }, { key: 'value', header: 'Valor', type: 'num', sortable: true, defaultSortDir: 'asc', card: 'primary' }];
const table = (props: Partial<React.ComponentProps<typeof DataTable<typeof data[number]>>> = {}) => <DataTable columns={columns} rows={data} rowKey={row => row.id} caption="Prueba" state="ready" {...props} />;
const rowNames = () => screen.getAllByRole('row').slice(1).map(row => row.querySelectorAll('[role="cell"]')[0]?.textContent?.replace(/^Nombre/, '').trim());

describe('DataTable', () => {
  afterEach(cleanup);
  it('cicla el orden ascendente, descendente y original mediante clics', () => {
    render(table());
    const button = screen.getByRole('button', { name: /Valor/ });
    const header = screen.getByRole('columnheader', { name: /Valor/ });
    fireEvent.click(button);
    expect(rowNames()).toEqual(['Beta', 'Gamma', 'Alfa']);
    expect(header.getAttribute('aria-sort')).toBe('ascending');
    fireEvent.click(button);
    expect(rowNames()).toEqual(['Alfa', 'Gamma', 'Beta']);
    expect(header.getAttribute('aria-sort')).toBe('descending');
    fireEvent.click(button);
    expect(rowNames()).toEqual(['Alfa', 'Beta', 'Gamma']);
    expect(header.getAttribute('aria-sort')).toBe('none');
  });
  it('cicla desde la dirección predeterminada descendente', () => {
    const descColumns = columns.map(column => column.key === 'value' ? { ...column, defaultSortDir: 'desc' as const } : column);
    render(<DataTable columns={descColumns} rows={data} rowKey={row => row.id} caption="Prueba" state="ready" />);
    const button = screen.getByRole('button', { name: /Valor/ });
    const header = screen.getByRole('columnheader', { name: /Valor/ });
    fireEvent.click(button);
    expect(rowNames()).toEqual(['Alfa', 'Gamma', 'Beta']);
    expect(header.getAttribute('aria-sort')).toBe('descending');
    fireEvent.click(button);
    expect(rowNames()).toEqual(['Beta', 'Gamma', 'Alfa']);
    expect(header.getAttribute('aria-sort')).toBe('ascending');
    fireEvent.click(button);
    expect(rowNames()).toEqual(['Alfa', 'Beta', 'Gamma']);
    expect(header.getAttribute('aria-sort')).toBe('none');
  });
  it('notifica el cambio de orden al controlador', () => {
    const onSortChange = vi.fn();
    render(table({ sort: null, onSortChange }));
    const button = screen.getByRole('button', { name: /Valor/ });
    fireEvent.click(button);
    expect(onSortChange).toHaveBeenLastCalledWith({ key: 'value', dir: 'asc' });
  });
  it.each(['loading', 'empty', 'error', 'stale'] as const)('muestra estado %s', state => {
    render(table({ state, staleAsOf: '14:05 ET' }));
    if (state === 'loading') expect(document.querySelector('.qa-skeleton')).toBeTruthy();
    if (state === 'empty') expect(screen.getByText('Sin datos')).toBeTruthy();
    if (state === 'error') expect(screen.getByText('No pudimos cargar los datos.')).toBeTruthy();
    if (state === 'stale') expect(screen.getByText(/Sin datos nuevos/)).toBeTruthy();
  });
  it('usa tarjetas y encabezados de grupo', () => {
    render(table({ cardBreakpoint: 2000, groupBy: row => ({ id: row.name[0], label: row.name[0] }) }));
    expect(document.querySelectorAll('.qa-datatable-group').length).toBe(3);
    expect(document.querySelector('.qa-card-title')).toBeTruthy();
  });
  it('ordena las tarjetas al cambiar el selector', () => {
    render(table({ cardBreakpoint: 2000 }));
    fireEvent.change(screen.getByRole('combobox', { name: 'Ordenar por' }), { target: { value: 'value' } });
    expect(rowNames()).toEqual(['Beta', 'Gamma', 'Alfa']);
  });
  it('usa color neutral para porcentajes en estado stale', () => {
    const pctColumns: Column<typeof data[number]>[] = [{ key: 'value', header: 'Cambio', type: 'pct' }];
    render(<DataTable columns={pctColumns} rows={data} rowKey={row => row.id} caption="Prueba" state="stale" />);
    expect(document.querySelector('.qa-pos, .qa-neg')).toBeNull();
    expect(document.querySelectorAll('.qa-neutral')).toHaveLength(3);
  });
});
