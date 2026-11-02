import { EquityChart, DivergingBar, Sparkline } from './Charts'
import { equity, sectorExposure } from '../mocks/data'
import { ThemeProvider } from '../ThemeContext'

export const CurvaDeValor = () => <ThemeProvider><EquityChart data={equity} /></ThemeProvider>
export const ExposicionPorSector = () => <ThemeProvider><DivergingBar data={sectorExposure} /></ThemeProvider>
export const Tendencia = () => <ThemeProvider><Sparkline values={[2, 4, 3, 7, 8]} label="Tendencia positiva" tone="positive" /></ThemeProvider>
