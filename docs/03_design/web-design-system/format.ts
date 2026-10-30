/**
 * Formato de datos es-AR para QuantAgent Web.
 * Reglas (hoja 02): USD 98.412,37 · −0,94 % (menos U+2212) · hora con zona siempre.
 */

const MINUS = '−';
const nf = (d: number) =>
  new Intl.NumberFormat('es-AR', { minimumFractionDigits: d, maximumFractionDigits: d });

/** Número con decimales fijos. Reemplaza el guion por el menos tipográfico. */
export function num(v: number, decimals = 2): string {
  return nf(decimals).format(v).replace('-', MINUS);
}

/** Número con signo explícito: +1.240,50 · −312,80 · 0,00 */
export function signed(v: number, decimals = 2): string {
  if (v > 0) return '+' + nf(decimals).format(v);
  if (v < 0) return MINUS + nf(decimals).format(Math.abs(v));
  return nf(decimals).format(0);
}

/** USD 98.412,37 · con signo: +USD 1.240,50 */
export function money(v: number, opts: { signed?: boolean; decimals?: number } = {}): string {
  const d = opts.decimals ?? 2;
  const body = nf(d).format(Math.abs(v));
  if (opts.signed) return (v > 0 ? '+' : v < 0 ? MINUS : '') + 'USD ' + body;
  return (v < 0 ? MINUS : '') + 'USD ' + body;
}

/** Compacto para tarjetas chicas: USD 98,4 k · USD 1,2 M */
export function moneyCompact(v: number): string {
  const a = Math.abs(v);
  const s = v < 0 ? MINUS : '';
  if (a >= 1e6) return `${s}USD ${nf(1).format(a / 1e6)} M`;
  if (a >= 1e3) return `${s}USD ${nf(1).format(a / 1e3)} k`;
  return `${s}USD ${nf(2).format(a)}`;
}

/** Porcentaje con signo: +1,28 % · −0,94 % (espacio normal antes del %) */
export function pct(v: number, decimals = 2, withSign = true): string {
  return (withSign ? signed(v, decimals) : num(v, decimals)) + ' %';
}

/** Puntos porcentuales: +6,2 pp */
export function pp(v: number, decimals = 1): string {
  return signed(v, decimals) + ' pp';
}

export type Direction = 'up' | 'down' | 'flat';
export const direction = (v: number, epsilon = 0): Direction =>
  v > epsilon ? 'up' : v < -epsilon ? 'down' : 'flat';

/* ---------- Fechas y horas ---------- */

export type Zone = 'ET' | 'ART';
const TZ: Record<Zone, string> = { ET: 'America/New_York', ART: 'America/Argentina/Buenos_Aires' };

/** 14:32 ET */
export function time(d: Date | string, zone: Zone): string {
  const t = new Intl.DateTimeFormat('es-AR', {
    hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TZ[zone],
  }).format(new Date(d));
  return `${t} ${zone}`;
}

/** 08:30 ET · 10:30 ART (eventos del calendario) */
export function timeBoth(d: Date | string): string {
  return `${time(d, 'ET')} · ${time(d, 'ART')}`;
}

/** jue 29 oct 2026 */
export function dateLong(d: Date | string, zone: Zone = 'ART'): string {
  return new Intl.DateTimeFormat('es-AR', {
    weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', timeZone: TZ[zone],
  }).format(new Date(d)).replace(/\./g, '').replace(/,/g, '');
}

/** 29/10 o 29/10/2026 */
export function dateShort(d: Date | string, withYear = false, zone: Zone = 'ART'): string {
  return new Intl.DateTimeFormat('es-AR', {
    day: '2-digit', month: '2-digit', ...(withYear ? { year: 'numeric' } : {}), timeZone: TZ[zone],
  }).format(new Date(d));
}

/** hace 4 min · hace 2 h · ayer · hace 3 días */
export function relative(d: Date | string, now: Date = new Date()): string {
  const s = Math.round((now.getTime() - new Date(d).getTime()) / 1000);
  if (s < 60) return 'recién';
  const m = Math.round(s / 60);
  if (m < 60) return `hace ${m} min`;
  const h = Math.round(m / 60);
  if (h < 24) return `hace ${h} h`;
  const days = Math.round(h / 24);
  return days === 1 ? 'ayer' : `hace ${days} días`;
}

/** Dato faltante: siempre raya + motivo aparte (nunca "0" ni "N/A"). */
export const MISSING = '—';
