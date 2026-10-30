import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { ThemeProvider, useTheme } from './ThemeContext'

function Preferences() {
  const { setTheme, setDensity } = useTheme()
  return (
    <>
      <button onClick={() => setTheme('dark')}>Dark</button>
      <button onClick={() => setDensity('compact')}>Compact</button>
    </>
  )
}

describe('App', () => {
  it('renders QuantAgent', () => {
    render(<ThemeProvider><App /></ThemeProvider>)
    expect(screen.getByText('QuantAgent')).toBeTruthy()
  })

  it('applies theme and density changes to the document', () => {
    render(<ThemeProvider><Preferences /></ThemeProvider>)
    fireEvent.click(screen.getByText('Dark'))
    fireEvent.click(screen.getByText('Compact'))
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.documentElement.dataset.density).toBe('compact')
  })

  it('renders when localStorage is unavailable', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    expect(() => render(<ThemeProvider><App /></ThemeProvider>)).not.toThrow()
    expect(screen.getByText('QuantAgent')).toBeTruthy()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    delete document.documentElement.dataset.theme
    delete document.documentElement.dataset.density
  })
})
