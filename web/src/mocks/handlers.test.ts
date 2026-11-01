import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { setupServer } from 'msw/node'
import { accountRefs, accounts, calendarEvents, feedEvents, homeStatus, marketClock, orders, positions } from './data'
import { handlers } from './handlers'
import { startMocks } from './mock-mode'

const server = setupServer(...handlers)
beforeAll(() => server.listen())

describe('MSW sample API data', () => {
  afterEach(() => server.resetHandlers())
  afterAll(() => server.close())

  it('returns each account response in the API contract shape', async () => {
    const list = await fetch('http://localhost/api/accounts').then((response) => response.json())
    expect(list).toEqual(accountRefs)
    const active = await fetch('http://localhost/api/accounts/paper-m4/summary').then((response) => response.json())
    const noData = await fetch('http://localhost/api/accounts/live-main/summary').then((response) => response.json())
    expect(active).toEqual(accounts[0])
    expect(noData).toEqual(accounts[2])
  })

  it('serves the dashboard feeds in backend response shapes', async () => {
    const [feed, calendar, status] = await Promise.all(['/api/feed', '/api/calendar?days=14', '/api/home/status'].map(path => fetch(`http://localhost${path}`).then(response => response.json())))
    expect(feed).toEqual(feedEvents)
    expect(feed.find((event: (typeof feedEvents)[number]) => event.type === 'order_rejected')?.group_count).toBe(2)
    expect(calendar).toEqual(calendarEvents)
    expect(status).toEqual(homeStatus)
  })

  it('returns typed positions and their total matches the active account summary', async () => {
    const result = await fetch('http://localhost/api/accounts/paper-m4/positions').then((response) => response.json())
    expect(result).toEqual(positions)
    expect(result.filter((position: (typeof positions)[number]) => position.side === 'long')
      .reduce((sum: number, position: (typeof positions)[number]) => sum + position.market_value, 0)).toBe(accounts[0].long_value)
    expect(result.filter((position: (typeof positions)[number]) => position.side === 'short')
      .reduce((sum: number, position: (typeof positions)[number]) => sum + position.market_value, 0)).toBe(accounts[0].short_value)
    expect(positions.some(({ side }) => side === 'short')).toBe(true)
  })

  it('returns all requested order statuses and market clock shape', async () => {
    const result = await fetch('http://localhost/api/accounts/paper-m4/orders').then((response) => response.json())
    const clock = await fetch('http://localhost/api/market/clock').then((response) => response.json())
    expect(result.map((order: { symbol: string }) => order.symbol)).toEqual(orders.map(order => order.symbol))
    expect(new Set(result.map((order: (typeof orders)[number]) => order.status))).toEqual(new Set(['filled', 'pending', 'rejected', 'cancelled', 'expired']))
    expect(clock).toEqual(marketClock)
  })

  it('does not register the worker unless mock=1 is present', () => {
    const start = vi.fn()
    expect(startMocks('', start)).toBe(false)
    expect(startMocks('?mock=0', start)).toBe(false)
    expect(start).not.toHaveBeenCalled()
    expect(startMocks('?mock=1', start)).toBe(true)
    expect(start).toHaveBeenCalledOnce()
  })
})
