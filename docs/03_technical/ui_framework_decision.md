# UI Framework Decision: Streamlit MVP → FastAPI+Angular Phase 2

## Decision Summary

**Phase 1 MVP**: **Streamlit** (Python only, fast development)
**Phase 2 Production**: **FastAPI** (API) + **Angular** (UI)

---

## Phase 1: Streamlit MVP

### Why Streamlit

**Advantages**:
- ✅ Develop in 3-4 days (not 2-3 weeks)
- ✅ Python only (no JavaScript needed)
- ✅ Perfect for data visualization
- ✅ Hot reload (changes reflect instantly)
- ✅ Great for internal tools & dashboards
- ✅ You know Python, not TypeScript

**Disadvantages**:
- ❌ Limited customization
- ❌ Looks "demo-ish" not production
- ❌ Slower for 1000s of rows
- ❌ Hard to share externally

### Streamlit MVP Dashboard

**File**: `quantagent/dashboard/streamlit_app.py`

```python
import streamlit as st
import pandas as pd
from quantagent.portfolio.manager import PortfolioManager
from quantagent.backtesting.backtest import Backtest

st.set_page_config(layout="wide", page_title="QuantAgent Trading System")
st.title("Trading System Dashboard")

# Load portfolio
portfolio = PortfolioManager.from_db()

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Backtest", "Trades", "Logs"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"${portfolio.get_total_value():,.2f}")
    col2.metric("Daily P&L", f"${portfolio.get_daily_pnl():,.2f}")
    col3.metric("Win Rate", "42%")

    st.line_chart(equity_curve)
    st.dataframe(recent_trades)

with tab2:
    st.subheader("Run Backtest")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")
    assets = st.multiselect("Assets", ["BTC", "SPX", "CL"])

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            backtest = Backtest(start_date, end_date, assets)
            results = backtest.run()

        st.metric("Win Rate", f"{results['win_rate']:.1%}")
        st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
        st.dataframe(results['trades'])

with tab3:
    st.dataframe(all_trades, use_container_width=True)

with tab4:
    st.dataframe(logs, use_container_width=True)

# Auto-refresh every 5 seconds
st.rerun()
```

**Run**:
```bash
streamlit run quantagent/dashboard/streamlit_app.py
```

**Accessible at**: `http://localhost:8501`

---

## Phase 2: FastAPI + Angular

### Migration Strategy

**Don't rewrite**, migrate incrementally:

1. **Phase 2.1**: Extract FastAPI backend
   - Create `quantagent/api/main.py` (FastAPI app)
   - Define routes: `/api/portfolio`, `/api/trades`, `/api/backtest`
   - Streamlit still works as fallback

2. **Phase 2.2**: Create Angular frontend
   - Build `ui/angular-app/`
   - Consume FastAPI endpoints
   - Streamlit still available

3. **Phase 2.3**: Switch to Angular as primary
   - Deprecate Streamlit
   - Keep FastAPI as main API

### FastAPI Backend

**File**: `quantagent/api/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from quantagent.portfolio.manager import PortfolioManager

app = FastAPI(title="QuantAgent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/portfolio")
def get_portfolio():
    portfolio = PortfolioManager.from_db()
    return {
        "cash": portfolio.cash,
        "positions": portfolio.positions,
        "total_value": portfolio.get_total_value(),
        "daily_pnl": portfolio.get_daily_pnl()
    }

@app.get("/api/trades")
def get_trades(limit: int = 100):
    trades = db.query(Trade).limit(limit).all()
    return [t.to_dict() for t in trades]

@app.post("/api/backtest")
def run_backtest(request: BacktestRequest):
    backtest = Backtest(
        request.start_date,
        request.end_date,
        request.assets
    )
    results = backtest.run()
    return results

@app.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    await websocket.accept()
    while True:
        portfolio = PortfolioManager.from_db()
        await websocket.send_json({
            "type": "portfolio_update",
            "data": portfolio.to_dict()
        })
        await asyncio.sleep(5)  # Update every 5 seconds
```

**Run**:
```bash
uvicorn quantagent.api.main:app --reload
```

**API at**: `http://localhost:8000/docs` (automatic Swagger docs)

### Angular Frontend

**File**: `ui/src/app/dashboard/dashboard.component.ts`

```typescript
@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  portfolio: Portfolio;
  trades: Trade[];
  equityCurve: ChartData;

  constructor(private api: ApiService, private ws: WebSocketService) {}

  ngOnInit() {
    // Subscribe to real-time portfolio updates
    this.ws.connect('/ws/portfolio').subscribe(update => {
      this.portfolio = update.data;
    });

    // Initial trades fetch
    this.api.getTrades(100).subscribe(trades => {
      this.trades = trades;
    });
  }

  runBacktest() {
    this.api.backtest(this.start_date, this.end_date, this.assets)
      .subscribe(results => {
        this.displayBacktestResults(results);
      });
  }
}
```

### Advantages Phase 2 UI

- ✅ Professional appearance
- ✅ Real-time WebSocket updates
- ✅ Fully customizable
- ✅ Scales to 10k+ trades
- ✅ Can share with team
- ✅ Mobile responsive (optional)
- ✅ Deployment-ready

---

## Comparison: Streamlit vs FastAPI+Angular

| Criteria | Streamlit | FastAPI+Angular |
|----------|-----------|---|
| **Development Speed** | 2-3 days | 2-3 weeks |
| **Learning Curve** | Easy (Python) | Medium (TypeScript) |
| **Customization** | Limited | Unlimited |
| **Real-time Updates** | Manual refresh | WebSocket native |
| **Professional Look** | No | Yes |
| **Scalability** | 100s of rows | 100ks of rows |
| **Shareability** | Limited (internal) | Unlimited |
| **Deployment** | Single Python app | API + UI separation |
| **Best for** | MVP | Production |

---

## Hybrid Approach

**You can have both**:

```
Phase 1:
└─ Streamlit MVP for validation

Phase 2:
├─ FastAPI backend (new)
├─ Angular frontend (new)
└─ Streamlit still works (optional fallback)
```

This way:
- ✅ Team familiar with Streamlit dashboard doesn't break
- ✅ New production UI runs in parallel
- ✅ Slow migration, no big bang rewrite

---

## Deployment

### Streamlit MVP
```bash
# Local
streamlit run quantagent/dashboard/streamlit_app.py

# Production (Streamlit Cloud, free tier available)
# Just push to GitHub, Streamlit Cloud deploys automatically
```

### FastAPI+Angular Phase 2
```bash
# Development
docker-compose up  # Runs both API and PostgreSQL

# Production
# API: Docker container on cloud (AWS, GCP, etc)
# UI: Static files on CDN (Cloudflare, AWS S3, etc)
```

---

## Recommendation Summary

| Phase | UI | Why |
|-------|----|----|
| **Phase 1 MVP** | Streamlit | Get working fast, validate strategy |
| **Phase 2 Prod** | FastAPI+Angular | Professional, scalable, maintainable |

This approach:
- ✅ Prioritizes validation over UI
- ✅ Delivers MVP fast (Week 10)
- ✅ Doesn't waste time on pretty UI that might change
- ✅ Allows professional UI once strategy proven

