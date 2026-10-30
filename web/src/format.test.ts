import { describe, expect, it } from 'vitest'
import { money, pct } from './lib/format'

describe('format es-AR', () => {
  it('formats money and percentage', () => {
    expect(money(98412.37)).toBe('USD 98.412,37')
    expect(pct(-0.94)).toBe('−0,94 %')
  })
})
