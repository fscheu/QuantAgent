from __future__ import annotations

import pandas as pd


def df_from_query(rows) -> pd.DataFrame:
    """Create a DataFrame from SQLAlchemy rows safely."""
    try:
        return pd.DataFrame([r.__dict__ for r in rows]).drop(columns=["_sa_instance_state"], errors="ignore")
    except Exception:
        return pd.DataFrame()

