from __future__ import annotations

import streamlit as st

from apps.streamlit.utils.ui import df_from_query


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
        st.checkbox("Only with order link", value=False, key="only_with_orders")

    if db.ok:
        with db.SessionLocal() as s:
            try:
                q = s.query(db.models.Signal)
                if symbol_filter:
                    q = q.filter(db.models.Signal.symbol.contains(symbol_filter))
                if timeframe_filter:
                    q = q.filter(db.models.Signal.timeframe.contains(timeframe_filter))
                if min_conf > 0:
                    q = q.filter(db.models.Signal.confidence >= min_conf)
                results = q.order_by(db.models.Signal.generated_at.desc()).limit(200).all()
                st.dataframe(df_from_query(results), use_container_width=True)
            except Exception as e:
                st.info(f"No signals or error reading signals: {e}")
    else:
        st.info("Connect DB to view analyses.")

