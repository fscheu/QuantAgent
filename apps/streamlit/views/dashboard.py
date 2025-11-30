from __future__ import annotations

from datetime import datetime

import streamlit as st

from apps.streamlit.utils.ui import df_from_query


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def render(db, environment: str) -> None:
    st.subheader("Dashboard")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_value = "-"
    daily_pnl = "-"
    win_rate = "-"
    open_positions = 0
    open_orders = 0

    if db.ok:
        SessionLocal = db.SessionLocal
        models = db.models
        with SessionLocal() as s:
            try:
                open_positions = s.query(models.Position).count()
                open_orders = s.query(models.Order).filter(models.Order.environment == environment).count()
                trades_query = s.query(models.Trade).filter(models.Trade.environment == environment)
                trades_total = trades_query.count()
                if trades_total:
                    wins = trades_query.filter(models.Trade.pnl.isnot(None)).filter(models.Trade.pnl > 0).count()
                    win_rate = f"{(wins / max(trades_total, 1)) * 100:.1f}%"

                today_start = datetime.combine(datetime.utcnow().date(), datetime.min.time())
                daily_trades = (
                    trades_query.filter(models.Trade.closed_at.isnot(None)).filter(models.Trade.closed_at >= today_start).all()
                )
                if daily_trades:
                    pnl_value = sum(_to_float(t.pnl) for t in daily_trades if t.pnl is not None)
                    daily_pnl = f"${pnl_value:,.2f}"

                positions = s.query(models.Position).all()
                portfolio_value = sum(_to_float(p.current_price) * _to_float(p.quantity) for p in positions)
                if portfolio_value:
                    total_value = f"${portfolio_value:,.2f}"
            except Exception:
                pass

    kpi1.metric("Portfolio Value", total_value)
    kpi2.metric("Daily P&L", daily_pnl)
    kpi3.metric("Win Rate", win_rate)
    kpi4.metric("Open Positions", open_positions)
    kpi5.metric("Open Orders", open_orders)

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Recent Trades**")
        if db.ok:
            with db.SessionLocal() as s:
                try:
                    q = (
                        s.query(db.models.Trade)
                        .filter(db.models.Trade.environment == environment)
                        .order_by(db.models.Trade.opened_at.desc())
                        .limit(50)
                        .all()
                    )
                    st.dataframe(df_from_query(q), use_container_width=True)
                except Exception as e:
                    st.info(f"No trades or error reading trades: {e}")
        else:
            st.info("Connect DB to view trades.")

    with colB:
        st.markdown("**Scheduler Status**")
        st.write("Status: unknown (MVP placeholder)")
        st.write("Next run: -  | Last run: -  | Errors: -")

