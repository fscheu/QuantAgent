import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { Inicio } from './Inicio'
import { handlers } from '../mocks/handlers'
import { calendarEvents, feedEvents, homeStatus } from '../mocks/data'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => { server.resetHandlers(); cleanup() })
afterAll(() => server.close())

describe('Inicio', () => {
  it('shows state banner severity and links it to the affected account', async () => {
    server.use(http.get('*/api/home/status', () => HttpResponse.json({ ...homeStatus, level: 'critical', headline: 'Atención: 1 alarma crítica' })))
    render(<Inicio navigate={() => undefined} />)
    const banner = await screen.findByText('Atención: 1 alarma crítica')
    expect(banner.closest('.qa-notice-critical')).toBeTruthy()
    expect(banner.closest('a')?.getAttribute('href')).toBe('/cuenta/paper-m4')
  })

  it('presents grouped feed items by severity and latest time, and sorts calendar dates', async () => {
    render(<Inicio navigate={() => undefined} />)
    const feed = await screen.findByRole('heading', { name: 'Desde tu última visita' }).then(node => node.parentElement!)
    await waitFor(() => expect(within(feed).getByText(/2 órdenes rechazadas/)).toBeTruthy())
    const feedTitles = [...feed.querySelectorAll('.qa-home-feed strong')].map(node => node.textContent)
    expect(feedTitles).toEqual(feedEvents.map(event => `${event.title}${event.group_count > 1 ? ` · ${event.group_count}` : ''}`))
    const calendar = screen.getByRole('heading', { name: 'Próximos eventos' }).parentElement!
    const dates = [...calendar.querySelectorAll('.qa-home-events time')].map(node => (node as HTMLTimeElement).dateTime)
    expect(dates).toEqual([...calendarEvents].sort((a, b) => a.date.localeCompare(b.date)).map(event => event.date))
    expect(dates[0]).toBe('2026-11-02')
  })
})
