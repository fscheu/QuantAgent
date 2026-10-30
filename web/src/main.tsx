import React from 'react'
import ReactDOM from 'react-dom/client'
import '@fontsource/geist-sans/400.css'
import '@fontsource/geist-sans/500.css'
import '@fontsource/geist-sans/600.css'
import '@fontsource/geist-mono/400.css'
import '@fontsource/geist-mono/500.css'
import { App } from './App'
import { ThemeProvider } from './ThemeContext'
import './styles/tokens.css'

if (import.meta.env.DEV) {
  const { startMocks } = await import('./mocks/worker')
  startMocks(window.location.search)
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
)
