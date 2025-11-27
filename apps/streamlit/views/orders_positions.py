from __future__ import annotations

import streamlit as st

from apps.streamlit.utils.ui import df_from_query


def render(db, environment: str) -> None:
    st.subheader("Orders & Positions (paper)")

    if db.ok:
        with db.SessionLocal() as s:
            try:
                st.markdown("**Orders**")
                q1 = s.query(db.models.Order).order_by(db.models.Order.created_at.desc()).limit(200).all()
                st.dataframe(df_from_query(q1), use_container_width=True)

                st.markdown("**Positions**")
                q2 = s.query(db.models.Position).order_by(db.models.Position.opened_at.desc()).limit(200).all()
                st.dataframe(df_from_query(q2), use_container_width=True)
            except Exception as e:
                st.info(f"No orders/positions or error reading: {e}")
    else:
        st.info("Connect DB to view orders & positions.")

