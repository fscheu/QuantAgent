import { ApiError } from '../api/client'
import { useCycleReceipt } from '../api/hooks'
import { ErrorState, Skeleton } from '../components'
import { CycleReceipt as ReceiptView } from '../components/CycleReceipt'
import { tableConfig } from '../config'

export function CycleReceiptScreen({ id, navigate }: { id: string; navigate: (path: string) => void }) {
  const receipt = useCycleReceipt(id)
  if (receipt.status === 'loading') return <Skeleton label="Recibo del ciclo" />
  if (receipt.status === 'error') return <ErrorState title={receipt.error instanceof ApiError && receipt.error.status === 404 ? 'No existe el ciclo' : undefined} message={receipt.error instanceof ApiError && receipt.error.status === 404 ? `No existe el ciclo ${id}.` : 'No pudimos cargar el recibo.'} action={{ label: 'Reintentar', onClick: receipt.reload }} />
  return <ReceiptView receipt={receipt.data} presentation={window.innerWidth >= tableConfig.cardBreakpoint ? 'panel' : 'screen'} onClose={() => navigate('/embudo')} />
}
