import { useEffect } from 'react'
import { ThemeProvider } from '../ThemeContext'
import { worker } from '../mocks/worker'
import { EtfDetail } from './EtfDetail'

function MockedEtfDetail() {
  useEffect(() => { void worker.start(); return () => { void worker.stop() } }, [])
  return <EtfDetail symbol="SPY" navigate={() => {}} />
}

export const ETF = () => <ThemeProvider><MockedEtfDetail /></ThemeProvider>
