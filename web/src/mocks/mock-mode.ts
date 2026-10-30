export function startMocks(search: string, startWorker: () => void): boolean {
  if (new URLSearchParams(search).get('mock') !== '1') return false
  startWorker()
  return true
}
