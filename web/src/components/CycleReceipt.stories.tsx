import type { Story } from '@ladle/react'
import { cycleReceipts } from '../mocks/data'
import { CycleReceipt } from './CycleReceipt'

export const Cerrado: Story = () => <CycleReceipt receipt={cycleReceipts[0]} onClose={() => {}} />
export const Parcial: Story = () => <CycleReceipt receipt={cycleReceipts[1]} onClose={() => {}} />
