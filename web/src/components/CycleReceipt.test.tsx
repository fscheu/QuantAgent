import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { cycleReceipts } from '../mocks/data'
import { CycleReceipt } from './CycleReceipt'

afterEach(cleanup)
describe('CycleReceipt', () => {
  it('muestra el hash corto, la medida y separa ERROR y ALARMA', () => {
    const receipt = { ...cycleReceipts[0], drops: [{ ...cycleReceipts[0].drops[0], verdict: 'ERROR' as const }, { ...cycleReceipts[0].drops[0], verdict: 'ALARMA' as const }] }
    render(<CycleReceipt receipt={receipt} onClose={() => {}} />)
    expect(screen.getByText('0123456789ab')).toBeTruthy()
    expect(screen.getAllByText(/0\.31 · umbral >= 0\.5/)).toHaveLength(2)
    expect(screen.getByLabelText('ERROR')).toBeTruthy()
    expect(screen.getByLabelText('ALARMA')).toBeTruthy()
  })
  it('identifica recibo parcial y fecha vacía', () => {
    render(<CycleReceipt receipt={cycleReceipts[1]} onClose={() => {}} />)
    expect(screen.getByText('parcial · todavía abierto')).toBeTruthy()
    expect(screen.getByText(/mercado.*—/)).toBeTruthy()
    expect(cycleReceipts[1]).toMatchObject({ partial: true, id: null, hash: null })
  })
})
