import { setupWorker } from 'msw/browser'
import { handlers } from './handlers'
import { startMocks as startWhenRequested } from './mock-mode'

export const worker = setupWorker(...handlers)

export function startMocks(search: string): boolean {
  return startWhenRequested(search, () => { void worker.start() })
}
