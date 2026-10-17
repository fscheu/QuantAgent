# QuantAgent-d4cq.47: Universo amplio de ETFs (2007)

Se versiona el manifiesto del snapshot `etf-1d-2026-10-amplio` en `tests/fixtures/snapshots/etf-1d-2026-10-amplio.json` para la cartera de primas de riesgo (D6, PR #252).

## Regla objetiva de inclusión

La primera sesión disponible en Yahoo Finance debe ser igual o anterior a `2007-01-03`.
Se conservan los 12 ETFs de `etf-1d-2026-10` y se incorporan 17 ETFs adicionales que cumplen la regla (29 activos en total).

## Composición por clase de activo (29 ETFs)

- **Acciones por región e índices (8):** SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000), EFA (EAFE), EEM (Emergentes), EWJ (Japón), EWZ (Brasil), FXI (China).
- **Acciones sectoriales US / Select Sector SPDR (9):** XLB (Materiales), XLE (Energía), XLF (Financiero), XLI (Industrial), XLK (Tecnología), XLP (Consumo Básico), XLU (Servicios Públicos), XLV (Salud), XLY (Consumo Discrecional).
- **Bonos / Renta Fija (6):** SHY (Tesoro 1-3y), IEF (Tesoro 7-10y), TLT (Tesoro 20y+), TIP (TIPS inflación), AGG (Agregado US), LQD (Corporativos Grado Inversión).
- **Materias primas (4):** DBC (Commodities diversificado), GLD (Oro), SLV (Plata), USO (Petróleo WTI).
- **Monedas (1):** FXE (Euro / USD).
- **Inmuebles (1):** VNQ (Vanguard Real Estate / REITs US).

## Candidatos excluidos

Candidatos que no cumplen la regla objetiva (primera vela posterior a `2007-01-03`):
- **DBA** (Invesco DB Agriculture): primera vela `2007-01-05`.
- **FXY** (CurrencyShares Japanese Yen): primera vela `2007-02-13`.
- **UUP** (DB US Dollar Bullish): primera vela `2007-03-01`.
- **HYG** (iShares High Yield Corporate Bond): primera vela `2007-04-11`.

## Aviso de supervivencia

Todos los ETFs seleccionados continúan cotizando hoy. El universo refleja los instrumentos que cotizaban en 2007 y han sobrevivido hasta la actualidad.
