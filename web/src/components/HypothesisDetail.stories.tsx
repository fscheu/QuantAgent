import type { Story } from '@ladle/react'
import { hypothesisDetail } from '../mocks/data'
import { AttemptList, StageTimeline } from './HypothesisDetail'
import '../styles/tokens.css'

const content = <div style={{ display: 'grid', gap: 'var(--qa-space-4)' }}><StageTimeline timeline={hypothesisDetail.timeline} /><AttemptList attempts={hypothesisDetail.attempts} /></div>
export const HypothesisDetailComponents: Story = () => content
HypothesisDetailComponents.meta = { title: 'Componentes / Ficha de hipótesis' }
export const HypothesisDetailComponentsDark: Story = () => <div data-theme="dark">{content}</div>
HypothesisDetailComponentsDark.meta = { title: 'Componentes / Ficha de hipótesis / Oscuro' }
