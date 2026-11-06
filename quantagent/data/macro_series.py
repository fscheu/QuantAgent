"""FRED macro series persistence and refresh logic."""

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
from sqlalchemy import select

from quantagent.models import MacroObservation


def update_macro_series(session, client, series=("T10Y2Y", "USREC"), start=None, dry_run=False):
    today = date.today()
    counts = {}
    for series_id in series:
        latest = session.scalar(
            select(MacroObservation.observation_date)
            .where(MacroObservation.series_id == series_id)
            .order_by(MacroObservation.observation_date.desc())
            .limit(1)
        )
        since = start or (latest - timedelta(days=7) if latest else date(2000, 1, 1))
        frame = client.observaciones(series_id, since, today, todas_las_versiones=False)
        count = 0
        for _, row in frame.iterrows():
            value = pd.to_numeric(row["valor"], errors="coerce")
            if pd.isna(value):
                continue
            observation_date = row["fecha"].date()
            value = Decimal(str(value))
            observation = session.get(MacroObservation, (series_id, observation_date))
            if observation is None:
                session.add(MacroObservation(
                    series_id=series_id,
                    observation_date=observation_date,
                    value=value,
                    published_at=today,
                ))
                count += 1
            elif observation.value != value:
                observation.value = value
        counts[series_id] = count
    session.flush()
    if dry_run:
        session.rollback()
    else:
        session.commit()
    return counts
