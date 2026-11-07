import { ApiError } from '../api/client'
import { useHypothesisDetail } from '../api/hooks'
import { ErrorState, Skeleton } from '../components'
import { HypothesisDetailContent } from '../components/HypothesisDetail'

export function Hipotesis({ id, navigate }: { navigate: (path: string) => void; id: string }) {
  const detail = useHypothesisDetail(id)
  if (detail.status === 'loading') return <main><Skeleton label="Hipótesis" /></main>
  if (detail.status === 'error') {
    const missing = detail.error instanceof ApiError && detail.error.status === 404
    return <main><ErrorState title={missing ? 'No existe la hipótesis' : undefined} message={missing ? 'No existe la hipótesis.' : 'No pudimos cargar la hipótesis.'} action={missing ? undefined : { label: 'Reintentar', onClick: detail.reload }} /></main>
  }
  return <main className="qa-hypothesis-detail"><HypothesisDetailContent hypothesis={detail.data} navigate={navigate} /></main>
}
