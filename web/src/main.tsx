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

const mockMode = import.meta.env.DEV || new URLSearchParams(window.location.search).get('mock') === '1'
const ready = mockMode
  ? import('./mocks/worker').then(({ startMocks }) => { startMocks(window.location.search) })
  : Promise.resolve()

void ready.then(() => ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
))
