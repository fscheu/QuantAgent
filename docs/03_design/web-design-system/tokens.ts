/**
 * QuantAgent Web — tokens en TypeScript.
 *
 * Para lo que NO puede leer variables CSS: lightweight-charts (canvas) y,
 * según la biblioteca, algunos gráficos de React. Mantener sincronizado con
 * tokens.css (mismos valores). Si se agrega un token de color, va en los dos.
 */

export type ThemeName = 'light' | 'dark';

export const palette = {
  light: {
    bg: '#F4F5F7', surface: '#FFFFFF', surfaceRaised: '#FFFFFF', surfaceSunken: '#EEF0F3',
    border: '#DDE1E6', borderStrong: '#C3C9D1', borderControl: '#848D9A',
    text: '#14181F', text2: '#4A5361', text3: '#646D7A',
    accent: '#0E6E68', onAccent: '#FFFFFF', accentSoft: '#E1F1EF',
    pos: '#1D5FC4', posSoft: '#E6EEFB', neg: '#B5420E', negSoft: '#FBEBE2',
    warn: '#8A5A00', warnSoft: '#FFF3D6', info: '#6342C9', infoSoft: '#EFEAFB',
    neutral: '#5B6472', neutralSoft: '#EDEFF2',
    grid: '#E8EBEF', axis: '#646D7A', recession: 'rgba(100,109,122,0.12)',
    candleUp: '#2FA36B', candleDown: '#B3261E', volume: '#C3C9D1',
  },
  dark: {
    bg: '#0E1116', surface: '#151A21', surfaceRaised: '#1C222B', surfaceSunken: '#0B0E12',
    border: '#29313C', borderStrong: '#3A4452', borderControl: '#6B7685',
    text: '#E8ECF1', text2: '#A9B2BE', text3: '#8A94A1',
    accent: '#3CC2B4', onAccent: '#06201E', accentSoft: '#12302E',
    pos: '#6EA4FF', posSoft: '#15233A', neg: '#FF8F5E', negSoft: '#3A1E12',
    warn: '#F2BE4E', warnSoft: '#33270C', info: '#B39CFF', infoSoft: '#241C3D',
    neutral: '#9AA4B1', neutralSoft: '#222933',
    grid: '#222933', axis: '#8A94A1', recession: 'rgba(154,164,177,0.12)',
    candleUp: '#3DCB8A', candleDown: '#E5484D', volume: '#3A4452',
  },
} as const;

/** Paleta categórica: asignar en este orden. Máximo 5 series a la vez. */
export const chartSeries: Record<ThemeName, readonly string[]> = {
  light: ['#2563C9', '#D9731A', '#0E8F86', '#9A4FD0', '#C93C72', '#6F7F1F', '#B08700', '#5E6C7C'],
  dark: ['#6EA4FF', '#FF9F4A', '#3CC2B4', '#C49BFF', '#FF7AAE', '#B5C55A', '#F2C94C', '#9AA8B8'],
};

/** Trazo por serie (además del color): nunca distinguir solo por color. */
export const seriesDash = ['none', '6 4', '2 4', '10 4 2 4', '1 0'] as const;

/** Escala divergente del mapa de calor (de −3 a +3 pasos). */
export const heatScale: Record<ThemeName, { bg: string; ink: string }[]> = {
  light: [
    { bg: '#B5420E', ink: '#FFFFFF' }, { bg: '#E58A5C', ink: '#14181F' }, { bg: '#F6CDB6', ink: '#14181F' },
    { bg: '#EDEFF2', ink: '#14181F' },
    { bg: '#C5D7F4', ink: '#14181F' }, { bg: '#6E99DE', ink: '#14181F' }, { bg: '#1D5FC4', ink: '#FFFFFF' },
  ],
  dark: [
    { bg: '#E0703F', ink: '#0E1116' }, { bg: '#9C4A26', ink: '#E8ECF1' }, { bg: '#4A2A1C', ink: '#E8ECF1' },
    { bg: '#222933', ink: '#E8ECF1' },
    { bg: '#1E3354', ink: '#E8ECF1' }, { bg: '#2F5FA8', ink: '#E8ECF1' }, { bg: '#6EA4FF', ink: '#0E1116' },
  ],
};

/** Paso del mapa de calor para un rendimiento en % (cortes: 0,25 · 0,75 · 1,5). */
export function heatStep(pct: number): number {
  if (pct <= -1.5) return 0;
  if (pct <= -0.75) return 1;
  if (pct < -0.25) return 2;
  if (pct <= 0.25) return 3;
  if (pct < 0.75) return 4;
  if (pct < 1.5) return 5;
  return 6;
}

/**
 * Opciones de lightweight-charts por tema.
 * Velas: alza = cuerpo hueco verde (upColor transparente), baja = lleno rojo.
 * Volumen: color de su vela al 35 %.
 */
export function candleTheme(theme: ThemeName) {
  const p = palette[theme];
  return {
    layout: {
      background: { color: p.surface },
      textColor: p.axis,
      fontFamily: "'Geist Mono', ui-monospace, monospace",
      fontSize: 11,
    },
    grid: { vertLines: { visible: false }, horzLines: { color: p.grid } },
    rightPriceScale: { borderVisible: false },
    timeScale: { borderVisible: false },
    crosshair: { vertLine: { color: p.text3 }, horzLine: { color: p.text3 } },
    candle: {
      upColor: 'rgba(0,0,0,0)',
      downColor: p.candleDown,
      borderUpColor: p.candleUp,
      borderDownColor: p.candleDown,
      wickUpColor: p.candleUp,
      wickDownColor: p.candleDown,
    },
    volumeUp: withAlpha(p.candleUp, 0.35),
    volumeDown: withAlpha(p.candleDown, 0.35),
    movingAverage: chartSeries[theme][3],
  };
}

function withAlpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export const motion = { fast: 120, base: 180, slow: 240, ease: 'cubic-bezier(0.2, 0, 0, 1)' } as const;

export const breakpoints = { tablet: 640, desktop: 1024 } as const;
