from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from apps.streamlit.utils.ui import df_from_query


def render(db) -> None:
    st.subheader("Logs")

    if not db.ok:
        st.info("Connect DB to view logs.")
        return

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_levels = st.multiselect(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=["INFO", "WARNING", "ERROR", "CRITICAL"],
        )

    with col2:
        symbol_filter = st.text_input("Symbol (contains)", value="")

    with col3:
        event_type_filter = st.text_input("Event Type (contains)", value="")

    with col4:
        hours_back = st.number_input(
            "Hours Back", min_value=1, max_value=168, value=24
        )

    # Query logs with filters
    with db.SessionLocal() as session:
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

            query = session.query(db.models.Log).filter(
                db.models.Log.timestamp >= cutoff_time
            )

            if selected_levels:
                query = query.filter(db.models.Log.level.in_(selected_levels))

            if symbol_filter:
                query = query.filter(
                    db.models.Log.symbol.ilike(f"%{symbol_filter}%")
                )

            if event_type_filter:
                query = query.filter(
                    db.models.Log.event_type.ilike(f"%{event_type_filter}%")
                )

            logs = (
                query.order_by(db.models.Log.timestamp.desc())
                .limit(500)
                .all()
            )

            if not logs:
                st.info("No logs found matching the filters.")
                return

            # Display logs as dataframe
            st.markdown(f"**Showing {len(logs)} log entries (max 500)**")
            df = df_from_query(logs)

            # Select relevant columns for display
            display_columns = [
                "timestamp",
                "level",
                "module",
                "event_type",
                "symbol",
                "message",
            ]
            available_columns = [
                col for col in display_columns if col in df.columns
            ]
            st.dataframe(df[available_columns], width="stretch")

            # Expandable details for recent 10 logs
            st.markdown("---")
            st.markdown("**Recent Log Details (expandable)**")

            for log in logs[:10]:
                expander_title = (
                    f"{log.timestamp} - {log.level} - {log.module}"
                )
                with st.expander(expander_title):
                    st.markdown(f"**Message:** {log.message}")
                    st.markdown(f"**Level:** {log.level}")
                    st.markdown(f"**Module:** {log.module}")
                    st.markdown(f"**Timestamp:** {log.timestamp}")
                    if log.symbol:
                        st.markdown(f"**Symbol:** {log.symbol}")
                    if log.event_type:
                        st.markdown(f"**Event Type:** {log.event_type}")
                    if log.environment:
                        st.markdown(f"**Environment:** {log.environment}")
                    if log.thread_id:
                        st.markdown(f"**Thread ID:** {log.thread_id}")
                    if log.checkpoint_id:
                        st.markdown(f"**Checkpoint ID:** {log.checkpoint_id}")
                    if log.extra_data:
                        st.markdown("**Metadata (JSON):**")
                        st.json(log.extra_data)

        except Exception as e:
            st.error(f"Error querying logs: {e}")
