import type { Story } from '@ladle/react';
import type { Funnel, ReviewRule, Verdict } from '../api/types';
import { CycleMilestones, DropReasons, FunnelChart, ReviewCounter, StageVerdictBadge } from './Funnel';
import '../styles/tokens.css';

const rule: ReviewRule = { limit: 6, streak: 4, reached: false, last_cycles: [{ cycle: '2026-06', promoted: 0 }, { cycle: '2026-07', promoted: 0 }, { cycle: '2026-08', promoted: 1 }, { cycle: '2026-09', promoted: 0 }, { cycle: '2026-10', promoted: 0 }, { cycle: '2026-11', promoted: 0 }] };
const funnel: Funnel = { cycle: 'all', period: 'all', total: 8, sample_too_small: false, promoted: 2, stages: (['e0', 'e1', 'e2', 'e3'] as const).map((stage, i) => ({ stage, reached: 8 - i * 2, here: 2, passed: 5 - i, cut: 2, error: i === 1 ? 1 : 0, alarm: i === 2 ? 1 : 0 })), later: [{ phase: 'shadow', reached: 2, here: 1 }, { phase: 'paper', reached: 1, here: 1 }, { phase: 'candidate', reached: 0, here: 0 }], drop_reasons: [{ stage: 'e1', verdict: 'CORTA', cause_code: 'sharpe_below', cause_label: 'Sharpe bajo el umbral', count: 2, example: { metric: 'Sharpe', value: 0.31, threshold: 0.5, unit: 'ratio', comparator: '>=' } }] };
const content = <div style={{ display: 'grid', gap: 'var(--qa-space-4)' }}><div style={{ display: 'flex', gap: 'var(--qa-space-2)' }}>{(['PASA', 'CORTA', 'ERROR', 'ALARMA'] as Verdict[]).map(verdict => <StageVerdictBadge key={verdict} verdict={verdict} />)}</div><FunnelChart funnel={funnel} /><DropReasons reasons={funnel.drop_reasons} /><ReviewCounter rule={rule} /><CycleMilestones milestones={null} cycle="2026-11" status="open" /></div>;
export const FunnelComponents: Story = () => content;
FunnelComponents.meta = { title: 'Componentes / Embudo / Claro' };
export const FunnelComponentsDark: Story = () => <div data-theme="dark">{content}</div>;
FunnelComponentsDark.meta = { title: 'Componentes / Embudo / Oscuro' };
