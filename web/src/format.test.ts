import { describe, expect, it } from 'vitest'
import { money, pct } from './lib/format'

describe('format es-AR', () => {
  it('formats money and percentage', () => {
    expect(money(98412.37)).toBe('USD 98.412,37')
    expect(pct(-0.0094)).toBe('−0,94 %')
    expect(pct(0.00029)).toBe('+0,03 %')
    expect(pct(0)).toBe('0,00 %')
    expect(pct(0.0128, 1, false)).toBe('1,3 %')
  })
})
