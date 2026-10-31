import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

type Theme = 'light' | 'dark'
type Density = 'comfortable' | 'compact'

type ThemeContextValue = {
  theme: Theme
  density: Density
  setTheme: (theme: Theme) => void
  setDensity: (density: Density) => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

function readPreference<T extends string>(key: string, values: readonly T[]): T | null {
  try {
    const value = localStorage.getItem(key)
    return values.find((option) => option === value) ?? null
  } catch {
    return null
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() =>
    readPreference('qa-theme', ['light', 'dark']) ??
      (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
  )
  const [density, setDensity] = useState<Density>(() =>
    readPreference('qa-density', ['comfortable', 'compact']) ?? 'comfortable',
  )

  useEffect(() => {
    const root = document.documentElement
    root.dataset.theme = theme
    root.dataset.density = density
    try {
      localStorage.setItem('qa-theme', theme)
      localStorage.setItem('qa-density', density)
    } catch {
      // Storage can be unavailable; the in-memory preferences still apply.
    }
  }, [theme, density])

  return (
    <ThemeContext.Provider value={{ theme, density, setTheme, setDensity }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) throw new Error('useTheme must be used within ThemeProvider')
  return context
}
