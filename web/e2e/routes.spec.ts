import { test, expect } from '@playwright/test'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { handlers } from '../src/mocks/handlers'
import { performance } from '../src/mocks/data'

const server = setupServer(...handlers,
  http.get('*/api/performance/:type/:id', ({ params }) => HttpResponse.json({
    ...performance, subject: { ...performance.subject, type: params.type, id: params.id },
  })),
)
test.beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
test.afterAll(() => server.close())

const routes = ['/', '/cuenta', '/cuenta/paper-m4', '/backtests',
  '/rendimiento/backtest/41', '/rendimiento/run/41', '/cuenta/paper-m4/rendimiento']
for (const width of [390, 1280]) {
  for (const path of routes) {
    test(`${path} a ${width}px`, async ({ page }, testInfo) => {
      const errors: string[] = []
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
      page.on('pageerror', error => errors.push(error.message))
      await page.setViewportSize({ width, height: 900 })
      await page.route('**/api/**', async route => {
        const response = await fetch(route.request().url())
        await route.fulfill({ status: response.status, contentType: 'application/json', body: await response.text() })
      })
      await page.goto(path)
      await page.waitForLoadState('networkidle')
      await page.evaluate(() => document.fonts.ready)
      await page.screenshot({ path: testInfo.outputPath('screen.png'), fullPage: true })
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width)
      expect(errors).toEqual([])
    })
  }
}
