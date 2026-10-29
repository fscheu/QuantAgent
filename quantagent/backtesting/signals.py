"""Puente estrategia -> columnas de señales por vela (QuantAgent-d4cq.3)."""

import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.macro import recortar_snapshot_macro
from ..data.snapshot import SNAPSHOT_DIR_ENV, snapshot_root
from ..models import ActivePosition, ExitPolicy, OrderSide
from ..strategy.base import TradingStrategy
from ..strategy.indicadores import atr

_DECISION = {"LONG": 1, "SHORT": -1}


def bar_signals(strategy: TradingStrategy, candles: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Señales de `strategy` por vela, sin que vea nunca velas posteriores a la que evalúa.

    `candles`: columnas timestamp, open, high, low, close, volume (las que recibe la estrategia).
    Devuelve, con el índice de `candles`: entry (+1 LONG, -1 SHORT, 0), exit (bool, solo la regla propia
    de should_exit), stop_loss_pct, take_profit_pct, trailing_stop_pct (NaN sin señal).
    Columnas de entrada stop: entry_order ('market' por defecto), entry_price (NaN) y entry_expiry_bars (NaN).
    Salida Chandelier: chandelier_dist (chandelier_mult × ATR(atr_period) de la ventana en cada vela si la
    estrategia define ambos atributos; NaN si no).
    """
    out = pd.DataFrame(
        {
            "entry": 0,
            "exit": False,
            "stop_loss_pct": np.nan,
            "take_profit_pct": np.nan,
            "trailing_stop_pct": np.nan,
            "entry_order": "market",
            "entry_price": np.nan,
            "entry_expiry_bars": np.nan,
            "chandelier_dist": np.nan,
        },
        index=candles.index,
    )
    n = strategy.required_history_bars
    if n <= 0:
        n = 30
    chand_mult = getattr(strategy, "chandelier_mult", None)
    atr_period = getattr(strategy, "atr_period", None)
    has_chand = chand_mult is not None and atr_period is not None
    col_chand = out.columns.get_loc("chandelier_dist")
    for i in range(len(candles)):
        window = candles.iloc[max(0, i + 1 - n): i + 1]
        price = float(window["close"].iloc[-1])
        if has_chand:
            atr_s = atr(window, n=int(atr_period))
            if len(atr_s) > 0 and pd.notna(atr_s.iloc[-1]):
                out.iloc[i, col_chand] = float(chand_mult * atr_s.iloc[-1])
        signal = strategy.generate_signal(window.to_dict(orient="records"), symbol, timeframe, price)
        if signal is not None and signal.decision in _DECISION:
            ref = signal.entry_price or price
            out.iloc[i, out.columns.get_loc("entry")] = _DECISION[signal.decision]
            if getattr(signal, "entry_order", None) is not None:
                out.iloc[i, out.columns.get_loc("entry_order")] = signal.entry_order
            if getattr(signal, "entry_price", None) is not None:
                out.iloc[i, out.columns.get_loc("entry_price")] = float(signal.entry_price)
            if getattr(signal, "entry_expiry_bars", None) is not None:
                out.iloc[i, out.columns.get_loc("entry_expiry_bars")] = float(signal.entry_expiry_bars)
            if signal.stop_loss is not None:
                out.iloc[i, out.columns.get_loc("stop_loss_pct")] = abs(ref - signal.stop_loss) / ref
            if signal.take_profit is not None:
                out.iloc[i, out.columns.get_loc("take_profit_pct")] = abs(signal.take_profit - ref) / ref
            if signal.trailing_stop_pct is not None:
                out.iloc[i, out.columns.get_loc("trailing_stop_pct")] = signal.trailing_stop_pct
        # Posición de prueba larga con stop y take profit inalcanzables: solo dispara la regla de la estrategia.
        probe = ActivePosition(
            symbol=symbol, side=OrderSide.BUY, entry_price=price, stop_loss=0.0, take_profit=1e12, quantity=1.0,
            decision_timestamp=datetime.min, candles_since_entry=0, exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3, candles_direction=[],
        )
        out.iloc[i, out.columns.get_loc("exit")] = bool(strategy.should_exit(probe, price, window)[0])
    return out


def check_no_lookahead(
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame], candles: pd.DataFrame, cuts, truncated_fn: Callable | None = None
) -> list[str]:
    """Prueba de truncamiento (QuantAgent-d4cq.4): lista vacía = `signal_fn` no mira el futuro.

    Para cada T de `cuts` compara `signal_fn(candles.loc[:T])` con `signal_fn(candles).loc[:T]`; `candles` debe
    tener índice de fechas. Si se pasa `truncated_fn(velas_hasta_T, T)`, reemplaza a `signal_fn` en el lado truncado. Devuelve un texto por cada (corte, columna) que difiere, con la primera fecha distinta.
    """
    full = signal_fn(candles)
    diffs = []
    for cut in cuts:
        truncated = truncated_fn(candles.loc[:cut], cut) if truncated_fn else signal_fn(candles.loc[:cut])
        expected = full.loc[:cut]
        for col in expected.columns:
            if col not in truncated.columns:
                diffs.append(f"{col}: falta en la señal truncada en {cut}")
                continue
            a, b = truncated[col], expected[col]
            same = (a == b) | (a.isna() & b.isna())
            if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
                same |= np.isclose(a, b, equal_nan=True)
            if not same.all():
                diffs.append(f"{col}: difiere con corte en {cut}, primera fecha {same.index[~same][0]}")
    return diffs


@contextmanager
def _snapshot_dir(path):
    previo = os.environ.get(SNAPSHOT_DIR_ENV)
    os.environ[SNAPSHOT_DIR_ENV] = str(path)
    try:
        yield
    finally:
        if previo is None:
            os.environ.pop(SNAPSHOT_DIR_ENV, None)
        else:
            os.environ[SNAPSHOT_DIR_ENV] = previo


def check_no_lookahead_macro(
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame], candles: pd.DataFrame, cuts, snapshot: str, base_dir=None
) -> list[str]:
    """`check_no_lookahead` que además recorta a T las versiones de `snapshot` (QuantAgent-d4cq.71).

    `signal_fn` tiene que leer el snapshot por nombre desde `$QUANTAGENT_SNAPSHOT_DIR` (como
    `serie_punto_en_el_tiempo`); en cada corte esa variable apunta a una copia recortada a T. Una estrategia
    que toma la última revisión en vez de lo publicado en la fecha ve valores distintos y falla.
    """
    raiz = snapshot_root(base_dir)
    with tempfile.TemporaryDirectory() as tmp, _snapshot_dir(raiz):

        def truncada(velas: pd.DataFrame, corte) -> pd.DataFrame:
            destino = Path(tmp) / f"{pd.Timestamp(corte):%Y%m%d%H%M%S}"
            recortar_snapshot_macro(snapshot, corte, destino, base_dir=raiz)
            with _snapshot_dir(destino):
                return signal_fn(velas)

        return check_no_lookahead(signal_fn, candles, cuts, truncated_fn=truncada)
