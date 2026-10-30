import { dateLong } from '../lib/format'

export function DateStepper({ value, onChange }: { value: string; onChange: (date: string) => void }) {
  const move = (days: number) => { const date = new Date(`${value}T12:00:00Z`); date.setUTCDate(date.getUTCDate() + days); onChange(date.toISOString().slice(0, 10)) }
  return <div className="qa-date-stepper"><button type="button" aria-label="Día anterior" onClick={() => move(-1)}>‹</button><label>Fecha <input aria-label="Fecha" type="date" value={value} onChange={event => onChange(event.target.value)} /> <span>{dateLong(`${value}T12:00:00Z`, 'ET')}</span></label><button type="button" aria-label="Día siguiente" onClick={() => move(1)}>›</button></div>
}
