<!-- Texto original de la propuesta de Fede (2026-10-25), copiado sin cambios de la conversación del proyecto.
     Es la fuente del diseño QuantAgent-M5. Las decisiones que lo modifican están en
     docs/03_design/QuantAgent-M5-DS-motor-top-down.md §1 y docs/02_planning/QuantAgent-M5-PL-motor-top-down.md. -->

# Especificación de Diseño y Análisis Funcional: Motor Jerárquico Macro-Fundamental y Ensamble Técnico en QuantAgent

## 1. Resumen Ejecutivo y Propósito
Este documento define la arquitectura e integración de una nueva rama de operación dentro de **QuantAgent** que combina lectura de régimen macroeconómico, selección fundamental seccional por sectores/países y ejecución mediante seguimiento de tendencias técnicas, preservando como principio rector la **esperanza matemática positiva (`E > 0`)** y el rigor estadístico fuera de muestra (*Out-of-Sample*).

### Veredicto Arquitectónico y Metodológico
1. **Unificación en el mismo sistema (`QuantAgent`):** Se descarta crear un sistema separado. Entre el 70% y el 80% de la infraestructura (ingesta y caché de datos, persistencia SQLAlchemy, validación determinista de riesgo, ruteo de órdenes hacia Alpaca, reconciliación de *fills* y auditoría de métricas) es idéntica. La nueva operatoria se implementa como un **Motor de Generación de Señales Federado (`HierarchicalMacroTrendEngine`)** que converge en un contrato único (`TradeIntent`) antes de la capa de riesgo.
2. **Alineación con Estándares Cuantitativos Institucionales (Anti-Overfitting):**
   - **En la Capa Macro (Etapa 1):** Se descarta la búsqueda por fuerza bruta de desfases temporales variables (*Lag Mining*), dado que los desfases macroeconómicos no son estacionarios y un error de 1 a 2 meses destruye el exceso de retorno. En su lugar, se adopta el enfoque institucional **Economic Trend + Time-Series Momentum (TSMOM)**: relaciones causales fijas definidas a priori (*Economic Priors*) combinadas 50/50 con el momentum de precios del sector y un filtro de protección de caídas (*Crash Avoidance Filter*).
   - **En la Capa Técnica (Etapa 3):** Se descarta optimizar qué estrategia o combinación de indicadores se adapta mejor a cada activo individual (*Unpooled Per-Asset Optimization*), práctica que infla artificialmente el Ratio de Sharpe en el backtest por sesgo de selección (*Selection Bias*) y degrada el desempeño fuera de muestra. En su lugar, se adopta **Asset-Class Pooling con Ensamble Fijo de Arquetipos Cerrados**, donde todos los activos operan bajo las mismas reglas estructurales y la adaptación a la microestructura de cada activo ocurre exclusivamente mediante **normalización por volatilidad ex-ante (`ATR` / `SafeZone`)**.

---

## 2. Registro de Decisiones de Arquitectura (ADRs)

| ID | Decisión | Qué se Comparte | Qué se Aísla | Justificación Técnica |
| :--- | :--- | :--- | :--- | :--- |
| **ADR-01** | **Core Compartido y Motores de Señal Federados** | Data Layer, Risk Gatekeeper, OMS/Alpaca Adapter, DB SQLAlchemy, Reportes. | Lógica de generación de señal (`QuantEngine` vs. `HierarchicalMacroTrendEngine`) y sus frecuencias de ejecución. | Evita duplicar infraestructura crítica y centraliza el control de riesgo de la cuenta. |
| **ADR-02** | **Contrato Único de Intención (`TradeIntent`)** | Estructura de entrada al `RiskGatekeeper` y al motor de ejecución. | El payload de trazabilidad (`metadata_json`) específico de cada motor. | Desacopla completamente el *por qué* se genera un trade de *cómo* se dimensiona, valida y ejecuta. |
| **ADR-03** | **Particionado de Capital en `PortfolioBuckets`** | Cuenta comitente única en el broker (`paper-alpaca` / live). | Asignación de capital virtual, límite de drawdown y *Portfolio Heat* por bucket. | Permite que las estrategias quant puras y la rama jerárquica macro convivan sin contaminar sus métricas ni pisar su margen. |
| **ADR-04** | **Matriz Macro Fija (*Economic Priors*) + TSMOM** | Conectores de datos de mercado y caché temporal. | Evaluación de régimen macroeconómico y ranking sectorial semanal/mensual. | Elimina regresiones espurias (Granger-Newbold) y el sobreajuste por pruebas múltiples en series macro cortas. |
| **ADR-05** | **Ensamble Técnico Agrupado (*Pooled*) + Sizing por Volatilidad** | `StrategyAssembler`, `TradingComponents` e indicadores existentes en `quantagent/`. | Parametrización dinámica basada únicamente en múltiplos de `ATR(14)` / `SafeZone` de cada activo. | Maximiza el tamaño muestral efectivo (`T_eff = T * K`) y evita el colapso *Out-of-Sample* del ajuste individual por activo. |
| **ADR-06** | **Compuerta de Auditoría (`OverfittingGate` + DSR)** | Motor de backtesting y registro de experimentos. | Validación *Point-in-Time* para datos macro/fundamentales y seguimiento en `paper-alpaca`. | Penaliza matemáticamente la cantidad de ensayos (`M`) mediante el *Deflated Sharpe Ratio* antes de habilitar despliegues a paper trading. |

---

## 3. Análisis Funcional del Pipeline Jerárquico (4 Etapas)

El flujo operativo funciona como un embudo asincrónico de cuatro etapas con **frecuencias de ejecución desacopladas** para optimizar cómputo y evitar ruido intradiario en decisiones estructurales.

### Etapa 1: Motor de Régimen Macro y Asignación Sectorial (`MacroRegimeEngine`)
* **Frecuencia de Ejecución:** Semanal (cierre de viernes) o Mensual.
* **Objetivo:** Determinar el sesgo general del mercado (*Risk-On*, *Neutral*, *Risk-Off*), el porcentaje máximo de exposición del bucket y habilitar los 2 o 3 sectores/países con mayor viento de cola estructural.
* **Requerimientos Funcionales:**
  1. **Ingesta *Point-in-Time*:** Todo dato macroeconómico debe indexarse por su fecha de publicación real (`release_date` / series *vintage*) y nunca por el período contable de referencia, evitando *look-ahead bias*.
  2. **Transformación de Estacionariedad:** Prohibido correlacionar o puntuar niveles nominales crudos. Toda serie macro se transforma a variación interanual (`YoY %`) o diferencia primera (`Delta X_t`) y luego se normaliza mediante un puntaje estandarizado móvil (*Rolling Z-Score* de 12 a 36 meses):
     `Z_t = (X_t - Media_[t-W, t]) / Desvio_[t-W, t]`
  3. **Diccionario de *Economic Priors* (5 Temas de *Economic Trend*):**
     - *Crecimiento:* ISM Manufacturing New Orders, Industrial Production (`INDPRO`), Initial Jobless Claims (invertido).
     - *Inflación:* CPI YoY, Breakeven Inflation 5Y/10Y (`T10YIE`).
     - *Política Monetaria y Curva:* Tasas Reales TIPS 10Y (`DFII10`), Pendiente 10Y-2Y (`T10Y2Y`).
     - *Condiciones Financieras y Riesgo:* High Yield Option-Adjusted Spread (`BAMLH0A0HYM2`), VIX, Índice Dólar (`DXY`).
  4. **Cálculo del Puntaje Sectorial Compuesto (`SectorScore`):**
     Cada ETF sectorial (`XLK`, `XLF`, `XLI`, `XLE`, `XLV`, `XLY`, `XLP`, `XLB`, `XLU`, `XLRE`, `XLC`) y ETF de país/región (`SPY`, `EWZ`, `EEM`, `ARGT`) recibe un puntaje compuesto:
     `SectorScore_i = 0.50 * MacroThemeScore_i + 0.50 * PriceMomentumScore_i`
     Donde `MacroThemeScore_i` surge de aplicar la matriz de signos fijos de los *Economic Priors* sobre los *Z-scores* macro actuales, y `PriceMomentumScore_i` mide el *Time-Series Momentum* (`TSMOM`) de 1, 3 y 6 meses ajustado por volatilidad.
  5. **Filtro de Protección (*Crash Avoidance Gate*):** Si el `PriceMomentumScore` del sector es negativo y el mercado amplio (`SPY`) opera debajo de su media estructural de largo plazo con pendiente negativa, el sector queda bloqueado para nuevas compras (`LONG`) independientemente de la lectura macro adelantada.

### Etapa 2: Screener Fundamental Seccional (`FundamentalFactorScreener`)
* **Frecuencia de Ejecución:** Semanal (sobre los sectores habilitados por la Etapa 1).
* **Objetivo:** Identificar dentro de cada sector habilitado los 3 a 5 activos líderes con mayor potencial de generación de *alpha* fundamental y acumulación institucional.
* **Requerimientos Funcionales:**
  1. **Normalización Seccional (*Cross-Sectional Z-Score*):** Todas las métricas fundamentales de un activo se normalizan estrictamente contra la media y el desvío estándar de los activos de **su mismo sector** en la fecha `t`.
  2. **Fórmula de Ranking Compuesto (`AssetFactorScore`):**
     - **Calidad y Rentabilidad (Peso 35%):** Retorno sobre el Capital Invertido (`ROIC`) y tendencia del Margen Operativo (`Operating Margin YoY diff`).
     - **Impulso Fundamental (Peso 35%):** Aceleración de ingresos trimestrales (`Revenue Growth QoQ acceleration`) y crecimiento/revisiones de beneficio por acción (`EPS Growth`).
     - **Fuerza Relativa Institucional (Peso 30%):** Desempeño relativo del activo frente al ETF de su propio sector en ventanas de 3 y 6 meses (`RS = Return_Asset / Return_Sector_ETF`). Se descartan activos en el cuartil inferior de fuerza relativa para evitar *value traps*.
  3. **Salida:** Persistencia de la tabla `CandidateAssets` conteniendo el Top 3 a 5 de activos por cada sector activo.

### Etapa 3: Ensamble Técnico Agrupado y Adaptativo por Volatilidad (`PooledTechnicalEnsemble`)
* **Frecuencia de Ejecución:** Diaria (al cierre de la vela diaria sobre el universo de `CandidateAssets`).
* **Objetivo:** Determinar el *timing* exacto de entrada, el nivel de *Stop Loss* inicial y los niveles de toma de ganancia o *Trailing Stop*, reutilizando `StrategyAssembler` y `TradingComponents`.
* **Requerimientos Funcionales:**
  1. **Prohibición de Hiperoptimización Individual:** No se optimizan períodos de indicadores activo por activo. Todos los activos candidatos son evaluados por un **Ensamble Fijo de Arquetipos Cerrados** validado a nivel de clúster (*Asset-Class Pooled*).
  2. **Arquetipos Técnicos del Ensamble:**
     - **Arquetipo A — *Pullback en Tendencia* (Basado en Triple Screen de Elder):**
       - *Pantalla 1 (Marea - Semanal):* Pendiente de EMA(26) semanal y dirección de MACD-Histograma semanal alcistas.
       - *Pantalla 2 (Ola - Diario):* Retroceso temporal medido por Force Index(2) debajo de cero o RSI(3) en zona de sobreventa (< 30) sin que el precio quiebre el soporte estructural ni la EMA(22) diaria.
       - *Pantalla 3 (Gatillo):* Orden `BUY_STOP` al máximo de la vela diaria previa más un *buffer* de ruido, con `Stop Loss` inicial calculado por indicador *SafeZone* o `2.0 * ATR(14)` debajo del mínimo del swing.
     - **Arquetipo B — *Breakout de Compresión de Volatilidad*:**
       - *Condición previa:* Compresión de volatilidad (*Bollinger Bandwidth* en percentil bajo de 60 ruedas) dentro de una tendencia de fondo alcista (`Precio > EMA(50) > EMA(200)`).
       - *Gatillo:* Cierre diario quebrando el máximo del canal de Donchian de 20 ruedas acompañado de volumen relativo mayor a `1.5x` el promedio de 20 ruedas.
       - *Stop Loss inicial:* `2.5 * ATR(14)` desde el punto de entrada o debajo del punto medio del rango de compresión.
  3. **Normalización por Volatilidad Ex-Ante:** Todas las distancias de filtro, objetivos y stops se expresan en múltiplos de `ATR(14)` del activo. Esto adapta automáticamente la holgura de la estrategia a la microestructura de volatilidad de cada instrumento sin agregar grados de libertad estadísticos.
  4. **Emisión de `TradeIntent`:** Cuando al menos un arquetipo dispara señal (o el puntaje combinado del ensamble supera el umbral de disparo), el motor construye y emite un DTO `TradeIntent`.

### Etapa 4: Compuerta de Riesgo Determinista y Seguimiento de Dos Velocidades (`RiskGatekeeper` & `PositionTracker`)
* **Frecuencia de Ejecución:** Pre-trade (en cada `TradeIntent`), Tiempo Real (en broker) y Diaria (post-cierre sobre posiciones abiertas).
* **Requerimientos Funcionales del `RiskGatekeeper` (100% Determinista):**
  Ningún agente ni motor de señal decide el tamaño de la posición. El `RiskGatekeeper` intercepta cada `TradeIntent` y aplica tres validaciones secuenciales:
  1. **Filtro de Asimetría Mínima (Reward/Risk):**
     `Reward_Risk_Ratio = abs(Target_Price - Entry_Price) / abs(Entry_Price - Stop_Loss_Price) >= R_min` (por defecto `R_min = 2.0`).
  2. **Cálculo de Tamaño de Posición (*Fixed Fractional Risk Sizing*):**
     `Max_Dollar_Risk = Bucket_Equity * Risk_Per_Trade_Pct` (ej. `1.0%` o `2.0%` del bucket).
     `Shares = floor(Max_Dollar_Risk / abs(Entry_Price - Stop_Loss_Price))`
     Además, se verifica el tope de concentración máxima por activo (ej. el valor nocional `Shares * Entry_Price` no puede superar el `20%` del `Bucket_Equity`).
  3. **Control de Riesgo Acumulado de Cartera (*Portfolio Heat* - Regla del 6%):**
     `Open_Portfolio_Heat = sum(Riesgo_Abierto_Posiciones_Activas) + Max_Dollar_Risk`
     Si `Open_Portfolio_Heat > (Bucket_Equity * Max_Bucket_Heat_Pct)` (por defecto `6.0%`), la orden es rechazada con estado `REJECTED_MAX_HEAT`.
* **Requerimientos Funcionales del `PositionTracker` (Dos Velocidades):**
  1. **Velocidad 1 (Protección Mecánica en Broker):** Toda orden de entrada ejecutada en Alpaca debe tener asociada en el servidor del broker su orden `STOP` física desde el instante `t = 0`.
  2. **Velocidad 2 (Gestión Táctica Diaria e Invalidación de Tesis):** Al cierre de cada rueda diaria, el `PositionTracker` evalúa las posiciones abiertas y puede emitir únicamente acciones que **reduzcan el riesgo** (prohibido alejar el Stop Loss original):
     - `HOLD`: Mantener Stop actual.
     - `MOVE_STOP_TO_BREAKEVEN`: Cuando la ganancia flotante supera `+1.5R`.
     - `TRAIL_STOP`: Actualizar el Stop Loss en Alpaca al nuevo valor del *SafeZone* / `Chandelier Exit (3 * ATR)` siempre que el nuevo valor sea superior al Stop vigente (en posiciones `LONG`).
     - `THESIS_INVALIDATION_TIGHTEN`: Si en el recálculo semanal de la Etapa 1 el sector pierde su estado alcista o pasa a *Risk-Off*, el sistema ajusta el Stop Loss al mínimo de las últimas 2 velas diarias o cierra la posición si ya está debajo de su media rápida (`EMA(10)`).

---

## 4. Contratos de Datos y Modelo de Persistencia (SQLAlchemy / DTOs)

### 4.1. Entidad `PortfolioBucket`
Permite particionar la cuenta de trading entre el motor quant clásico y el motor jerárquico macro.
```python
class PortfolioBucketModel(Base):
    __tablename__ = "portfolio_buckets"
    bucket_id: str          # PK, ej. "QUANT_CORE", "MACRO_HIERARCHICAL"
    allocation_pct: float   # Ej. 0.70 y 0.30
    allocated_equity: float
    max_risk_per_trade_pct: float  # Ej. 0.015 (1.5%)
    max_portfolio_heat_pct: float  # Ej. 0.060 (6.0%)
    is_active: bool
```

### 4.2. Contrato `TradeIntent` (DTO de Entrada al `RiskGatekeeper`)
```python
@dataclass(frozen=True)
class TradeIntent:
    intent_id: str
    timestamp_utc: datetime
    bucket_id: str                  # "QUANT_CORE" | "MACRO_HIERARCHICAL"
    strategy_id: str                # Ej. "ENSEMBLE_TRIPLE_SCREEN_A"
    symbol: str
    direction: Literal["LONG", "SHORT"]
    timeframe: str                  # "1D"
    order_type: Literal["STOP", "LIMIT", "MARKET"]
    entry_price: float
    stop_loss_price: float          # Obligatorio
    take_profit_price: float        # Obligatorio para validación de R/R
    atr_at_signal: float
    invalidation_rules: dict        # Ej. {"max_bars_unfilled": 3, "require_sector_bullish": True}
    reasoning_metadata: dict        # Snapshot de SectorScore, AssetFactorScore y valores de indicadores
```

### 4.3. Auditoría Estadística (`OverfittingGate` & `DeflatedSharpeRatio`)
Para evitar que estrategias sobreajustadas pasen a `paper-alpaca`, el módulo de backtesting debe registrar el número acumulado de variantes probadas (`M`) sobre un mismo conjunto de datos y calcular el **Deflated Sharpe Ratio (Bailey & López de Prado, 2014)**:
- Registra en base de datos cada ejecución de backtest (`trial_id`, `strategy_family`, `sharpe_ratio`, `skewness`, `kurtosis`, `sample_length_T`).
- Calcula el Sharpe máximo esperado por azar `E[max(SR)_M]` para `M` ensayos independientes y exige `DSR_p_value >= 0.95` junto con validación agrupada (*Asset-Class Pooled*) antes de habilitar el flag `approved_for_paper = True`.

---

## 5. Plan de Implementación y Desglose de Tickets (Backlog para Claude Code)

Se estructura el desarrollo en **4 Épicas incrementales** con entregables aislados y testeables:

### Épica 1: Desacoplamiento Core, Buckets y `RiskGatekeeper` Determinista
* **Ticket 1.1 — Modelos de Persistencia `PortfolioBucket` y `TradeIntentRecord`:**
  - *Alcance:* Crear modelos SQLAlchemy y migraciones para gestionar sub-carteras (`PortfolioBucket`) y persistir todas las intenciones de trade (`TradeIntentRecord`) con su `reasoning_metadata` y estado de aprobación/rechazo.
  - *Criterio de Aceptación:* Tests unitarios verificando creación de buckets y trazabilidad completa de un `TradeIntent` aprobado vs. rechazado por riesgo.
* **Ticket 1.2 — Servicio `RiskGatekeeper` (Sizing 2% y Portfolio Heat 6%):**
  - *Alcance:* Implementar el validador determinista que recibe un `TradeIntent` y el estado actual del `PortfolioBucket`, verifica `Reward/Risk >= R_min`, calcula `Shares` exactas por riesgo fraccional fijo y valida que el riesgo abierto total no supere el límite del 6%.
  - *Criterio de Aceptación:* Suite de tests parametrizados cubriendo rechazo por R/R insuficiente, tope de exposición nocional y bloqueo por exceso de *Portfolio Heat*.
* **Ticket 1.3 — Servicio `PositionTracker` de Dos Velocidades:**
  - *Alcance:* Implementar la lógica de seguimiento diario sobre posiciones abiertas en Alpaca que evalúa reglas de *Trailing Stop* (`SafeZone` / `ATR`), pasaje a *breakeven* e invalidación de tesis, garantizando por invariante de código que el nuevo `stop_loss_price` nunca amplíe el riesgo respecto al stop vigente.
  - *Criterio de Aceptación:* Tests de integración con mock de Alpaca verificando reemplazo de órdenes `STOP` y rechazo de cualquier intento de bajar el stop en una posición `LONG`.

### Épica 2: Ensamble Técnico Agrupado y Compuerta Anti-Overfitting (Etapa 3)
* **Ticket 2.1 — Arquetipos Técnicos Normalizados por `ATR` en `StrategyAssembler`:**
  - *Alcance:* Codificar los dos arquetipos cerrados (`ArchetypePullbackElder` y `ArchetypeVolatilityBreakout`) dentro de `quantagent/` reutilizando `TradingComponents`, parametrizando entradas, filtros y stops exclusivamente en unidades de `ATR(14)` / `SafeZone`.
  - *Criterio de Aceptación:* Ejecución limpia de ambos arquetipos en el motor de backtesting actual emitiendo objetos `TradeIntent` válidos.
* **Ticket 2.2 — Evaluador *Asset-Class Pooled* y Cálculo de `DeflatedSharpeRatio`:**
  - *Alcance:* Crear el módulo de validación estadística que corre el ensamble sin cambiar parámetros sobre una cesta homogénea de activos, calcula la esperanza matemática en múltiplos de riesgo (`R-Expectancy`) y el `DeflatedSharpeRatio` (ajustado por asimetría, curtosis y número de ensayos `M`).
  - *Criterio de Aceptación:* Función `calculate_deflated_sharpe_ratio()` validada con tests matemáticos contra los casos de referencia del paper de Bailey & López de Prado (2014).

### Épica 3: Motor Macro *Economic Trend* y Filtro de Régimen (Etapa 1)
* **Ticket 3.1 — Conector `MacroSeriesProvider` con Alineación *Point-in-Time*:**
  - *Alcance:* Implementar el proveedor de series macroeconómicas (FRED API / ETFs sectoriales y países) con soporte de caché local, indexación por `release_date` y transformaciones automáticas de estacionariedad (`YoY %`, diferencias primeras y *Rolling Z-Score*).
  - *Criterio de Aceptación:* Test de verificación *Point-in-Time* que garantice que al consultar una fecha histórica `t`, ninguna serie devuelva valores publicados en `t + k`.
* **Ticket 3.2 — Calculador `MacroRegimeEngine` (`SectorScore` + `CrashFilter`):**
  - *Alcance:* Implementar la matriz de signos fijos (*Economic Priors*) sobre los 5 temas macroeconómicos, combinarla 50/50 con el *Time-Series Momentum* (`TSMOM` 1M, 3M, 6M) de los ETFs sectoriales/países y aplicar la compuerta de bloqueo (`Crash Avoidance Gate`) cuando la tendencia estructural sea bajista.
  - *Criterio de Aceptación:* Generación y persistencia semanal del objeto `MarketRegimeState` con el ranking de sectores habilitados y bloqueados.

### Épica 4: Screener Fundamental Seccional y Orquestación Top-Down (Etapa 2 + Integración)
* **Ticket 4.1 — Módulo `FundamentalFactorScreener` (*Cross-Sectional Z-Score*):**
  - *Alcance:* Implementar el filtro fundamental que toma los componentes de los sectores habilitados en la Etapa 1 y calcula el puntaje normalizado intra-sectorial combinando Calidad (`ROIC`, Margen Operativo - 35%), Impulso de Ganancias (`Revenue/EPS acceleration` - 35%) y Fuerza Relativa frente al ETF del sector (`RS 3M/6M` - 30%).
  - *Criterio de Aceptación:* Selección determinista del Top `N` de activos por sector y exclusión automática de activos en el cuartil inferior de fuerza relativa.
* **Ticket 4.2 — Job Orquestador `HierarchicalMacroPipeline` e Integración con `paper-alpaca`:**
  - *Alcance:* Conectar las 4 etapas en el flujo programado del sistema: ejecución semanal de Etapas 1 y 2 (actualizando `CandidateAssets`) y ejecución diaria de Etapas 3 y 4 (generando `TradeIntents`, validando en `RiskGatekeeper` y despachando al entorno `paper-alpaca`).
  - *Criterio de Aceptación:* Test de integración de punta a punta simulando un ciclo semanal + diario completo desde la lectura macro hasta el envío de la orden con Stop Loss a `paper-alpaca`.

---

## 6. Referencias Metodológicas y Técnicas
1. **AQR Capital Management (Brooks, Ooi, Feilbogen & Akant, 2023):** *Economic Trend* — [https://www.aqr.com/Insights/Research/White-Papers/Economic-Trend](https://www.aqr.com/Insights/Research/White-Papers/Economic-Trend)
2. **Bailey, D. H. & López de Prado, M. (2014):** *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality* — [http://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf](http://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
3. **Carver, R. (2026):** *Pooling Analysis: Unpooled vs. Asset-Class Pooled Systematic Rules* — [https://qoppac.blogspot.com/2026/](https://qoppac.blogspot.com/2026/)
4. **Harvey, C. R., Liu, Y. & Zhu, H. (2016):** *… and the Cross-Section of Expected Returns* (*Review of Financial Studies*, criterio `t-stat > 3.0`) — [https://academic.oup.com/rfs/article-abstract/29/1/5/1843824](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824)
5. **St. Louis Fed (FRED / ALFRED API):** *Point-in-Time Macroeconomic Vintage Data* — [https://fred.stlouisfed.org/docs/api/fred/](https://fred.stlouisfed.org/docs/api/fred/)
