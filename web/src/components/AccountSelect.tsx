import type { AccountSummary } from '../api/types'

export function AccountSelect({ accounts, value, onChange }: { accounts: AccountSummary[]; value: string; onChange: (id: string) => void }) {
  return <label className="qa-account-select">Cuenta <select aria-label="Cuenta" value={value} onChange={event => onChange(event.target.value)}>{accounts.map(account => <option key={account.id} value={account.id}>{account.name}{account.status === 'no_data' ? ' · Sin datos todavía' : ''}</option>)}</select></label>
}
