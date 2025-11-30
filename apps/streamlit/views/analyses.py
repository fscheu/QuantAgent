from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st


def render(db, environment: str) -> None:
    st.subheader("Analyses – Explore Signals with Provenance")
    colF1, colF2, colF3, colF4 = st.columns(4)
    with colF1:
        symbol_filter = st.text_input("Symbol contains", value="")
    with colF2:
        timeframe_filter = st.text_input("Timeframe", value="")
    with colF3:
        min_conf = st.number_input("Min confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with colF4:
        order_link = st.checkbox("Only with order link", value=False, key="only_with_orders")

    colF5, colF6, colF7 = st.columns(3)
    with colF5:
        provider_filter = st.text_input("Model provider", value="")
    with colF6:
        model_filter = st.text_input("Model name", value="")
    with colF7:
        days_back = st.number_input("Days back", min_value=0, max_value=90, value=7, step=1)

    if db.ok:
        with db.SessionLocal() as s:
            try:
                q = s.query(db.models.Signal).filter(db.models.Signal.environment == environment)
                if symbol_filter:
                    q = q.filter(db.models.Signal.symbol.contains(symbol_filter))
                if timeframe_filter:
                    q = q.filter(db.models.Signal.timeframe.contains(timeframe_filter))
                if min_conf > 0:
                    q = q.filter(db.models.Signal.confidence >= min_conf)
                if order_link:
                    q = q.filter(db.models.Signal.order_id.isnot(None))
                if provider_filter:
                    q = q.filter(db.models.Signal.model_provider.contains(provider_filter))
                if model_filter:
                    q = q.filter(db.models.Signal.model_name.contains(model_filter))
                if days_back:
                    window_start = datetime.utcnow() - timedelta(days=int(days_back))
                    q = q.filter(db.models.Signal.generated_at >= window_start)
                results = q.order_by(db.models.Signal.generated_at.desc()).limit(200).all()

                if not results:
                    st.info("No signals found for the selected filters.")
                    return

                table_rows = []
                for r in results:
                    table_rows.append(
                        {
                            "generated_at": r.generated_at,
                            "symbol": r.symbol,
                            "timeframe": r.timeframe,
                            "signal": r.signal.value if hasattr(r.signal, "value") else r.signal,
                            "confidence": r.confidence,
                            "model_provider": r.model_provider,
                            "model_name": r.model_name,
                            "agent_version": r.agent_version,
                            "environment": r.environment,
                            "thread_id": r.thread_id,
                            "checkpoint_id": r.checkpoint_id,
                            "order_id": r.order_id,
                        }
                    )
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                st.markdown("**Details (latest 5)**")
                for detail in results[:5]:
                    with st.expander(f"{detail.symbol} {detail.timeframe} @ {detail.generated_at}"):
                        st.write("Signal:", detail.signal)
                        st.write("Confidence:", detail.confidence)
                        st.write("Model:", detail.model_provider, detail.model_name)
                        st.write("Thread/Checkpoint:", detail.thread_id, detail.checkpoint_id)
                        if detail.analysis_summary:
                            st.markdown("**Analysis summary**")
                            st.write(detail.analysis_summary)
                        st.markdown("**Indicators & patterns**")
                        st.json(
                            {
                                "rsi": detail.rsi,
                                "macd": detail.macd,
                                "stochastic": detail.stochastic,
                                "roc": detail.roc,
                                "williams_r": detail.williams_r,
                                "pattern": detail.pattern,
                                "trend": detail.trend,
                            }
                        )
            except Exception as e:
                st.info(f"No signals or error reading signals: {e}")
    else:
        st.info("Connect DB to view analyses.")

