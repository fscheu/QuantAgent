import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import App from './App'

test('renderiza <App /> y comprueba que aparece el texto QuantAgent', () => {
  render(<App />)
  expect(screen.getByText('QuantAgent')).toBeDefined()
})
