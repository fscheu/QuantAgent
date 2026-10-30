import type { AccountSummary, Order, Position } from './types'

export class ApiError extends Error {
  constructor(readonly status: number, message: string) { super(message); this.name = 'ApiError' }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) throw new ApiError(response.status, `Error ${response.status}`)
  return response.json() as Promise<T>
}

export const getAccounts = () => get<AccountSummary[]>('/api/accounts')
export const getSummary = (accountId: string) => get<AccountSummary>(`/api/accounts/${encodeURIComponent(accountId)}/summary`)
export const getPositions = (accountId: string) => get<Position[]>(`/api/accounts/${encodeURIComponent(accountId)}/positions`)
export const getOrders = (accountId: string, date: string) => get<Order[]>(`/api/accounts/${encodeURIComponent(accountId)}/orders?date=${encodeURIComponent(date)}`)
