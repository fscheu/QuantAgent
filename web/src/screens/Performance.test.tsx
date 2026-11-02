import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MetricGroup, MonthlyHeatmap, ShortSampleBanner } from './Performance'
import { performance } from '../mocks/data'
import { CumulativeChart, DrawdownChart, ReturnsBars, ReturnsHistogram, RollingChart } from '../components/PerformanceCharts'

describe('rendimiento de backtest', () => {
 afterEach(cleanup)
 it('muestra una métrica no calculable con motivo y sin valor numérico', () => {
  render(<MetricGroup title="Rendimiento" metrics={ [['cagr', performance.metrics.cagr]] } />)
  expect(screen.getByText('no calculable')).toBeTruthy()
  expect(screen.getByText('Necesita 12 meses; hay 3,6')).toBeTruthy()
  expect(screen.queryByText('0.0 %')).toBeNull()
 })
 it('muestra el banner de muestra corta', () => {
  render(<ShortSampleBanner report={performance}/>); expect(screen.getByText(/Muestra corta/)).toBeTruthy()
 })
 it('ofrece la tabla con los valores devueltos por la API', () => {
  render(<MonthlyHeatmap report={performance}/>); fireEvent.click(screen.getByRole('button',{name:'Tabla'}))
  expect(screen.getByRole('table').textContent).toContain('2.026,00'); expect(screen.getByRole('table').textContent).toContain('8,40'); expect(screen.getByRole('table').textContent).toContain('—')
 })
 it('no muestra números de SPY cuando el benchmark no está disponible', () => {
  render(<MonthlyHeatmap report={performance}/>); fireEvent.click(screen.getByRole('button',{name:'Tabla'}))
  expect(screen.queryByText('0.0 %')).toBeNull()
  expect(screen.getByRole('table').textContent).toContain('—')
 })
 it('cada gráfico ofrece una tabla real con los datos del mock', () => {
  const cases=[[<CumulativeChart report={performance}/>, '2026-01-01'],[<DrawdownChart report={performance}/>, '2026-01-01'],[<ReturnsBars report={performance}/>, '2026'],[<ReturnsHistogram report={performance}/>, '2,00'],[<RollingChart report={performance}/>, '2026-02-01']] as const
  for(const [chart,value] of cases){const {container,unmount}=render(chart);fireEvent.click(within(container).getByRole('button',{name:'Tabla'}));expect(within(container).getByRole('table').textContent).toContain(value);unmount()}
 })
})
