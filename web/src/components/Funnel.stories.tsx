import type { Story } from '@ladle/react';
import type { ReviewRule, Verdict } from '../api/types';
import { CycleMilestones, ReviewCounter, StageVerdictBadge } from './Funnel';
import '../styles/tokens.css';

const rule: ReviewRule = { limit: 6, streak: 4, reached: false, last_cycles: [{ cycle: '2026-06', promoted: 0 }, { cycle: '2026-07', promoted: 0 }, { cycle: '2026-08', promoted: 1 }, { cycle: '2026-09', promoted: 0 }, { cycle: '2026-10', promoted: 0 }, { cycle: '2026-11', promoted: 0 }] };
const content = <div style={{ display: 'grid', gap: 'var(--qa-space-4)' }}><div style={{ display: 'flex', gap: 'var(--qa-space-2)' }}>{(['PASA', 'CORTA', 'ERROR', 'ALARMA'] as Verdict[]).map(verdict => <StageVerdictBadge key={verdict} verdict={verdict} />)}</div><ReviewCounter rule={rule} /><CycleMilestones milestones={null} cycle="2026-11" status="open" /></div>;
export const FunnelComponents: Story = () => content;
FunnelComponents.meta = { title: 'Componentes / Embudo / Claro' };
export const FunnelComponentsDark: Story = () => <div data-theme="dark">{content}</div>;
FunnelComponentsDark.meta = { title: 'Componentes / Embudo / Oscuro' };
