import { useCallback, useEffect, useRef, useState } from 'react'
import { dataConfig } from '../config'
import * as api from './client'

export type LoadState<T> = { status: 'loading' } | { status: 'ok'; data: T } | { status: 'error'; error: Error }

export function useLoad<T>(load: () => Promise<T>, key: string): LoadState<T> & { reload: () => void } {
  const loadRef = useRef(load)
  loadRef.current = load
  const [state, setState] = useState<LoadState<T>>({ status: 'loading' })
  const [version, setVersion] = useState(0)
  const reload = useCallback(() => setVersion(value => value + 1), [])
  useEffect(() => {
    let active = true
    const run = () => loadRef.current().then(data => { if (active) setState({ status: 'ok', data }) }).catch(error => { if (active) setState({ status: 'error', error: error instanceof Error ? error : new Error(String(error)) }) })
    void run()
    const timer = window.setInterval(() => { void run() }, dataConfig.pollSecondsMarketOpen * 1000)
    return () => { active = false; window.clearInterval(timer) }
  }, [key, version])
  return { ...state, reload }
}

export const useAccounts = () => useLoad(api.getAccounts, 'accounts')
export const useSummary = (accountId: string) => useLoad(() => api.getSummary(accountId), `summary:${accountId}`)
export const usePositions = (accountId: string) => useLoad(() => api.getPositions(accountId), `positions:${accountId}`)
export const useEquity = (accountId: string, range: api.EquityRange) => useLoad(() => api.getEquity(accountId, range), `equity:${accountId}:${range}`)
export const useDividends = (accountId: string) => useLoad(() => api.getDividends(accountId), `dividends:${accountId}`)
export const useSectorExposure = (accountId: string) => useLoad(() => api.getSectorExposure(accountId), `sector-exposure:${accountId}`)
export const useOrders = (accountId: string, date: string) => useLoad(() => api.getOrders(accountId, date), `orders:${accountId}:${date}`)
export const useFeed = () => useLoad(api.getFeed, 'feed')
export const useCalendar = (days: number) => useLoad(() => api.getCalendar(days), `calendar:${days}`)
export const useHomeStatus = () => useLoad(api.getHomeStatus, 'home-status')
export const useFunnel = (cycle: string, period: string) => useLoad(() => api.getFunnel(cycle, period), `funnel:${cycle}:${period}`)
export const useCycles = () => useLoad(api.getCycles, 'funnel-cycles')
export const useCurrentCycle = () => useLoad(api.getCurrentCycle, 'current-cycle')
export const useReviewRule = () => useLoad(api.getReviewRule, 'review-rule')
export const useHistory = () => useLoad(api.getHistory, 'funnel-history')
export const useHypotheses = (filters: { cycle?: string; period?: string }) => useLoad(() => api.getHypotheses(filters), `hypotheses:${filters.cycle ?? ''}:${filters.period ?? ''}`)

export const useBacktests = () => useLoad(api.getBacktests, 'backtests')
export const useBacktestPerformance = (id: string, range: import('./types').PerfRange, vs: string | null) => useLoad(() => api.getBacktestPerformance(id, range, vs), `performance:${id}:${range}:${vs}`)
export const useAccountPerformance = (type: 'portfolio' | 'run', id: string, range: import('./types').PerfRange) => useLoad(() => api.getAccountPerformance(type, id, range), `performance:${type}:${id}:${range}`)
export const usePerformance = (type: 'backtest' | 'portfolio' | 'run', id: string, range: import('./types').PerfRange, vs: string | null) => useLoad(() => type === 'backtest' ? api.getBacktestPerformance(id, range, vs) : api.getAccountPerformance(type, id, range), `performance:${type}:${id}:${range}:${vs}`)
