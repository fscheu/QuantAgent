import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { money, num, pp } from '../lib/format';
import { tableConfig } from '../config';
import { EmptyState, ErrorState, Num, Skeleton } from './index';
import './DataTable.css';

export type Align = 'left' | 'right';
export type CardRole = 'title' | 'subtitle' | 'primary' | 'secondary' | 'hidden';
export interface Column<T> {
  key: string; header: string; type: 'text' | 'num' | 'money' | 'pct' | 'pp' | 'date' | 'badge' | 'custom';
  align?: Align; width?: string; sortable?: boolean; sortValue?: (row: T) => number | string;
  defaultSortDir?: 'asc' | 'desc'; render?: (row: T) => ReactNode; card?: CardRole;
}
export interface DataTableProps<T> {
  columns: Column<T>[]; rows: T[]; rowKey: (row: T) => string; caption: string;
  sort?: { key: string; dir: 'asc' | 'desc' } | null; onSortChange?: (sort: { key: string; dir: 'asc' | 'desc' } | null) => void;
  subRow?: (row: T) => ReactNode | null; groupBy?: (row: T) => { id: string; label: string; tone?: 'neg' | 'neutral' };
  density?: 'comfortable' | 'compact'; state: 'ready' | 'loading' | 'empty' | 'error' | 'stale'; staleAsOf?: string;
  emptyTitle?: string; emptyMessage?: string; emptyAction?: { label: string; onClick: () => void };
  errorMessage?: string; onRetry?: () => void; onRowClick?: (row: T) => void; cardBreakpoint?: number;
}

export function DataTable<T>({ columns, rows, rowKey, caption, sort, onSortChange, subRow, groupBy, density = 'comfortable', state, staleAsOf, emptyTitle = 'Sin datos', emptyMessage = 'No hay filas para mostrar.', emptyAction, errorMessage = 'No pudimos cargar los datos.', onRetry, onRowClick, cardBreakpoint = tableConfig.cardBreakpoint }: DataTableProps<T>) {
  const [localSort, setLocalSort] = useState<{ key: string; dir: 'asc' | 'desc' } | null>(null);
  const [mobile, setMobile] = useState(typeof window !== 'undefined' && window.innerWidth < cardBreakpoint);
  const activeSort = sort === undefined ? localSort : sort;
  const updateSort = (next: typeof activeSort) => { if (sort === undefined) setLocalSort(next); onSortChange?.(next); };
  const sorted = useMemo(() => {
    if (!activeSort) return rows;
    const column = columns.find(item => item.key === activeSort.key);
    if (!column) return rows;
    return [...rows].sort((a, b) => {
      const av = column.sortValue ? column.sortValue(a) : (a as Record<string, unknown>)[column.key] as number | string;
      const bv = column.sortValue ? column.sortValue(b) : (b as Record<string, unknown>)[column.key] as number | string;
      return (av < bv ? -1 : av > bv ? 1 : 0) * (activeSort.dir === 'asc' ? 1 : -1);
    });
  }, [rows, columns, activeSort]);
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const query = window.matchMedia(`(max-width: ${cardBreakpoint - 1}px)`);
    const change = () => setMobile(query.matches);
    query.addEventListener('change', change); return () => query.removeEventListener('change', change);
  });
  const clickSort = (column: Column<T>) => {
    if (!column.sortable) return;
    if (activeSort?.key !== column.key) updateSort({ key: column.key, dir: column.defaultSortDir ?? 'asc' });
    else if (activeSort.dir === (column.defaultSortDir ?? 'asc')) updateSort({ key: column.key, dir: activeSort.dir === 'asc' ? 'desc' : 'asc' });
    else updateSort(null);
  };
  const sortableColumns = columns.filter(column => column.sortable);
  const selectedSortColumn = sortableColumns.find(column => column.key === activeSort?.key) ?? sortableColumns[0];
  const value = (row: T, column: Column<T>) => {
    if (column.render) return column.render(row);
    const raw = (row as Record<string, unknown>)[column.key];
    if (typeof raw === 'number') {
      if (column.type === 'num') return <Num value={raw} />;
      if (column.type === 'money') return <span className="qa-num">{money(raw)}</span>;
      if (column.type === 'pct') return <span className={`qa-num ${state === 'stale' ? 'qa-neutral' : raw > 0 ? 'qa-pos' : raw < 0 ? 'qa-neg' : 'qa-neutral'}`}>{raw > 0 ? '▲ +' : raw < 0 ? '▼ −' : '— '}{num(Math.abs(raw))}</span>;
      if (column.type === 'pp') return <span className="qa-num">{pp(raw)}</span>;
      return column.type === 'date' ? String(raw) : num(raw);
    }
    return raw == null ? '—' : String(raw);
  };
  const groups = new Map<string, { label: string; tone?: string; rows: T[] }>();
  sorted.forEach(row => { const group = groupBy?.(row) ?? { id: '', label: '' }; const entry = groups.get(group.id) ?? { label: group.label, tone: group.tone, rows: [] }; entry.rows.push(row); groups.set(group.id, entry); });
  const renderRow = (row: T) => <div className="qa-datatable-row" role="row" key={rowKey(row)} style={{ gridTemplateColumns: columns.map(column => column.width ?? 'minmax(0, 1fr)').join(' ') }} tabIndex={onRowClick ? 0 : undefined} onClick={() => onRowClick?.(row)} onKeyDown={event => { if (event.key === 'Enter') onRowClick?.(row); }}>
    {columns.map(column => <div role="cell" key={column.key} className={`qa-datatable-cell qa-card-${column.card ?? 'secondary'} qa-align-${column.align ?? (['num', 'money', 'pct', 'pp'].includes(column.type) ? 'right' : 'left')}`} style={{ width: column.width }}>{mobile ? <><span className="qa-datatable-label">{column.header}</span><span>{value(row, column)}</span></> : value(row, column)}</div>)}
    {subRow?.(row) && <div className="qa-datatable-subrow">{subRow(row)}</div>}
  </div>;
  return <section className={`qa-datatable qa-density-${density}${state === 'stale' ? ' qa-datatable-stale' : ''}`} style={{ '--qa-table-card-breakpoint': `${cardBreakpoint}px` } as React.CSSProperties}>
    {state === 'stale' && <div className="qa-datatable-stale-note" role="status">Sin datos nuevos{staleAsOf && ` · ${staleAsOf}`}</div>}
    {mobile && sortableColumns.length > 0 && <div className="qa-datatable-card-sort"><label>Ordenar por: <select aria-label="Ordenar por" value={selectedSortColumn?.key ?? ''} onChange={event => { const column = sortableColumns.find(item => item.key === event.target.value); if (column) updateSort({ key: column.key, dir: column.defaultSortDir ?? 'asc' }); }}>{sortableColumns.map(column => <option key={column.key} value={column.key}>{column.header}</option>)}</select></label>{selectedSortColumn && <button type="button" onClick={() => updateSort({ key: selectedSortColumn.key, dir: activeSort?.key === selectedSortColumn.key && activeSort.dir === 'desc' ? 'asc' : 'desc' })}>{activeSort?.key === selectedSortColumn.key && activeSort.dir === 'desc' ? 'Mayor' : 'Menor'}</button>}</div>}
    <div className="qa-datatable-table" role="table" aria-label={caption}>
      <div className="qa-datatable-caption">{caption}</div>
      <div className="qa-datatable-head" role="row" style={{ gridTemplateColumns: columns.map(column => column.width ?? 'minmax(0, 1fr)').join(' ') }}>{columns.map(column => <div role="columnheader" key={column.key} aria-sort={!column.sortable ? undefined : activeSort?.key === column.key ? activeSort.dir === 'asc' ? 'ascending' : 'descending' : 'none'} className={`qa-align-${column.align ?? (['num', 'money', 'pct', 'pp'].includes(column.type) ? 'right' : 'left')}`}>{column.sortable ? <button type="button" onClick={() => clickSort(column)}>{column.header}<span aria-hidden="true">{activeSort?.key === column.key ? activeSort.dir === 'asc' ? ' ↑' : ' ↓' : ' ↕'}</span></button> : column.header}</div>)}</div>
      {state === 'loading' ? <div role="rowgroup">{Array.from({ length: tableConfig.skeletonRows }, (_, i) => <div role="row" className="qa-datatable-row" key={i}>{columns.map(column => <div role="cell" key={column.key}><Skeleton /></div>)}</div>)}</div> : state === 'empty' ? <EmptyState title={emptyTitle} message={emptyMessage} action={emptyAction} /> : state === 'error' ? <ErrorState message={errorMessage} action={onRetry ? { label: 'Reintentar ahora', onClick: onRetry } : undefined} /> : <div role="rowgroup">{Array.from(groups.entries()).map(([id, group]) => <div key={id || 'all'}>{groupBy && <div className={`qa-datatable-group qa-group-${group.tone ?? 'neutral'}`} role="heading">{group.label} · {group.rows.length}</div>}{group.rows.map(renderRow)}</div>)}</div>}
    </div>
  </section>;
}
