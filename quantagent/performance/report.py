"""Build a daily, benchmark-relative performance report without I/O."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

METRICS = (
    "total_return", "cagr", "positive_months", "best_month", "worst_month", "volatility",
    "sharpe", "sortino", "max_drawdown", "calmar", "alpha", "beta", "relative_return",
)
DEFAULT_THRESHOLDS = {
    "descriptive_days": 21, "risk_days": 63, "annualized_days": 252, "min_rebalances": 12,
    "positive_months": 3, "month_extremes": 2, "positive_months_short": 12,
    "month_extremes_short": 12, "risk_short_days": 252,
}


def _curve(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        result = values.copy()
    elif isinstance(values, pd.DataFrame):
        date_col = "date" if "date" in values else "timestamp"
        value_col = "equity" if "equity" in values else "value"
        result = pd.Series(values[value_col].to_numpy(), index=values[date_col])
    elif isinstance(values, Mapping):
        result = pd.Series(values)
    else:
        rows = list(values)
        date_key = "date" if rows and "date" in rows[0] else "timestamp"
        value_key = "equity" if rows and "equity" in rows[0] else "value"
        result = pd.Series([row[value_key] for row in rows], index=[row[date_key] for row in rows])
    result.index = pd.to_datetime(result.index).normalize()
    return result.astype(float).sort_index()


def _mv(value: float | None, status: str = "ok", reason: str | None = None, detail: dict | None = None) -> dict:
    result = {"value": None if value is None or not np.isfinite(value) else float(value), "status": status, "reason": reason}
    if detail is not None:
        result["detail"] = detail
    return result


def build_report(
    equity: Any,
    benchmark: Any,
    *,
    subject: Mapping[str, Any],
    range: str,
    align_with: Any = None,
    thresholds: Mapping[str, int] | None = None,
) -> dict:
    """Calculate a PerformanceReport from supplied equity and SPY observations."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    original = _curve(equity).dropna()
    spy = _curve(benchmark).dropna() if benchmark is not None and len(benchmark) else pd.Series(dtype=float)
    has_bench = len(spy) > 0
    if not has_bench:
        spy = pd.Series(1.0, index=original.index)  # neutro: se descarta al armar la salida
    if align_with is not None:
        other = _curve(align_with).dropna()
        common = original.index.intersection(spy.index).intersection(other.index)
        if len(common):
            common_start, common_end = common[0], common[-1]
            original, spy = original.loc[common_start:common_end], spy.loc[common_start:common_end]
    joined = pd.concat([original.rename("value"), spy.rename("benchmark")], axis=1).dropna()
    if len(joined) and range != "all":
        last = joined.index[-1]
        cut = {"ytd": pd.Timestamp(year=last.year, month=1, day=1)}.get(
            range, last - pd.DateOffset(years=int(range[:-1])) if range[:-1].isdigit() else joined.index[0])
        joined = joined.loc[joined.index >= cut]
    if len(joined):
        start, end = joined.index[0], joined.index[-1]
    else:
        start = end = pd.Timestamp("1970-01-01")
    values, bench = joined.value, joined.benchmark
    returns, breturns = values.pct_change().dropna(), bench.pct_change().dropna()
    returns, breturns = returns.align(breturns, join="inner")
    n = len(returns)
    total = float((1 + returns).prod() - 1) if n else 0.0
    btotal = float((1 + breturns).prod() - 1) if n else 0.0
    dd = values / values.cummax() - 1 if len(values) else values
    bdd = bench / bench.cummax() - 1 if len(bench) else bench
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_idx = dd.idxmin() if len(dd) else start
    peaks = values.loc[:max_idx]
    peak_idx = peaks[peaks == peaks.max()].index[-1] if len(peaks) else max_idx
    post_peak = values.loc[max_idx:]
    recovery_candidates = post_peak[post_peak >= values.loc[peak_idx]]
    recovery_idx = recovery_candidates.index[0] if len(recovery_candidates) else None

    month_rows = []
    monthly = pd.DataFrame({"a": values.groupby(values.index.to_period("M")).last(),
                            "b": bench.groupby(bench.index.to_period("M")).last()})
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    for date, row in monthly.iterrows():
        month_start = date.to_period("M").start_time
        complete = start <= month_start and end >= date.normalize()
        prior_a, prior_b = values.loc[:month_start - pd.Timedelta(days=1)], bench.loc[:month_start - pd.Timedelta(days=1)]
        if not complete or not len(prior_a) or not len(prior_b) or pd.isna(row.a) or pd.isna(row.b):
            continue
        month_rows.append({"year": int(date.year), "month": int(date.month), "ret": float((row.a / prior_a.iloc[-1] - 1) * 100),
                           "benchmark_ret": float((row.b / prior_b.iloc[-1] - 1) * 100), "complete": True})
    mvals = [r["ret"] for r in month_rows]
    annual_rows = []
    if len(values):
        for year, group in values.groupby(values.index.year):
            mask = bench.index.year == year
            bg = bench.loc[mask]
            annual_rows.append({"year": int(year), "ret": float((group.iloc[-1] / group.iloc[0] - 1) * 100),
                                "benchmark_ret": float((bg.iloc[-1] / bg.iloc[0] - 1) * 100) if len(bg) else 0.0,
                                "partial": year == start.year or year == end.year})

    rf = float(limits.get("risk_free_rate", 0.02))
    daily_rf = rf / 252
    sd = float(returns.std(ddof=1)) if n >= 2 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(returns.to_numpy() - daily_rf, 0) ** 2))) if n else 0.0
    cagr = (1 + total) ** (252 / n) - 1 if n and total > -1 else None
    var_b = float(breturns.var(ddof=1)) if n >= 2 else 0.0
    beta = float(returns.cov(breturns) / var_b) if var_b > 0 else None
    alpha = float((returns - daily_rf - (beta or 0) * (breturns - daily_rf)).mean() * 252) if n >= 2 and beta is not None else None
    positive_min = limits["positive_months"]
    month_min = limits["month_extremes"]
    values_by_key = {
        "total_return": total * 100, "cagr": cagr * 100 if cagr is not None else None,
        "positive_months": sum(x > 0 for x in mvals) / len(mvals) * 100 if len(mvals) >= positive_min else None,
        "best_month": max(mvals) if len(mvals) >= month_min else None,
        "worst_month": min(mvals) if len(mvals) >= month_min else None,
        "volatility": sd * np.sqrt(252) * 100 if n >= limits["risk_days"] else None,
        "sharpe": (returns.mean() - daily_rf) / sd * np.sqrt(252) if n >= limits["risk_days"] and sd else None,
        "sortino": (returns.mean() - daily_rf) / downside * np.sqrt(252) if n >= limits["risk_days"] and downside else None,
        "max_drawdown": max_dd * 100, "calmar": cagr / abs(max_dd) if cagr is not None and max_dd else None,
        "alpha": alpha * 100 if alpha is not None and n >= limits["annualized_days"] else None,
        "beta": beta if n >= limits["risk_days"] else None,
        "relative_return": ((1 + total) / (1 + btotal) - 1) * 100 if n else None,
    }
    minimums = {"total_return": limits["descriptive_days"], "cagr": limits["annualized_days"],
                "max_drawdown": limits["descriptive_days"], "relative_return": limits["descriptive_days"],
                "calmar": limits["annualized_days"], "alpha": limits["annualized_days"], "beta": limits["risk_days"],
                "volatility": limits["risk_days"], "sharpe": limits["risk_days"], "sortino": limits["risk_days"]}
    reasons = {"few_days"} if n < limits["descriptive_days"] else set()
    if len(mvals) < positive_min or len(mvals) < month_min:
        reasons.add("few_months")
    metrics = {}
    for key in METRICS:
        value = values_by_key[key]
        threshold = minimums.get(key, limits["positive_months"] if key == "positive_months" else limits["month_extremes"])
        sample_size = len(mvals) if key in ("positive_months", "best_month", "worst_month") else n
        if sample_size < threshold:
            value = None
        if value is None:
            metrics[key] = _mv(None, "not_computable", f"Necesita {threshold} observaciones; hay {n if key not in ('positive_months', 'best_month', 'worst_month') else len(mvals)}")
        else:
            short = (key in ("volatility", "sharpe", "sortino", "beta") and n < limits["risk_short_days"] or
                     key in ("positive_months", "best_month", "worst_month") and len(mvals) < limits["positive_months_short"])
            metrics[key] = _mv(value, "short_sample" if short else "ok")
    bm = {
        "total_return": btotal * 100,
        "cagr": ((1 + btotal) ** (252 / n) - 1) * 100 if n and btotal > -1 else None,
        "positive_months": sum(r["benchmark_ret"] > 0 for r in month_rows) / len(month_rows) * 100 if len(month_rows) >= positive_min else None,
        "best_month": max((r["benchmark_ret"] for r in month_rows), default=None) if len(month_rows) >= month_min else None,
        "worst_month": min((r["benchmark_ret"] for r in month_rows), default=None) if len(month_rows) >= month_min else None,
        "volatility": float(breturns.std(ddof=1) * np.sqrt(252) * 100) if n >= limits["risk_days"] else None,
        "sharpe": float((breturns.mean() - daily_rf) / breturns.std(ddof=1) * np.sqrt(252)) if n >= limits["risk_days"] and breturns.std(ddof=1) else None,
        "sortino": float((breturns.mean() - daily_rf) / np.sqrt(np.mean(np.minimum(breturns.to_numpy() - daily_rf, 0) ** 2)) * np.sqrt(252)) if n >= limits["risk_days"] and np.sqrt(np.mean(np.minimum(breturns.to_numpy() - daily_rf, 0) ** 2)) else None,
        "max_drawdown": float(bdd.min() * 100) if len(bdd) else None,
        "calmar": ((1 + btotal) ** (252 / n) - 1) / abs(float(bdd.min())) if n >= limits["annualized_days"] and len(bdd) and bdd.min() else None,
        "alpha": 0.0 if n >= limits["annualized_days"] else None,
        "beta": 1.0 if n >= limits["risk_days"] else None,
        "relative_return": 0.0 if n else None,
    }
    benchmark_metrics = {}
    for key, value in bm.items():
        benchmark_metrics[key] = _mv(value, "not_computable" if value is None else
                                     "short_sample" if (key in ("volatility", "sharpe", "sortino", "beta") and n < limits["risk_short_days"] or
                                                          key in ("positive_months", "best_month", "worst_month") and len(mvals) < limits["positive_months_short"]) else "ok",
                                     f"Necesita muestras suficientes; hay {n}" if value is None else None)
    complete_months = len(month_rows)
    monthly_returns = month_rows
    daily_dist = None
    daily_reason = None
    if n >= limits["risk_days"]:
        q = float(returns.quantile(.05))
        daily_bins = []
        daily_a, daily_b = returns.to_numpy() * 100, breturns.to_numpy() * 100
        edges = np.arange(np.floor(min(daily_a.min(), daily_b.min())) - 1, np.ceil(max(daily_a.max(), daily_b.max())) + 2)
        counts_a, _ = np.histogram(daily_a, bins=edges)
        counts_b, _ = np.histogram(daily_b, bins=edges)
        daily_bins = [{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(counts_a[i]),
                       "benchmark_count": int(counts_b[i])} for i in builtins.range(len(counts_a))]
        daily_dist = {"bin_width_pct": 1.0, "bins": daily_bins, "mean": float(returns.mean() * 100), "benchmark_mean": float(breturns.mean() * 100),
                      "n": n, "var_95": q * 100, "cvar_95": float(returns[returns <= q].mean() * 100)}
    else:
        daily_reason = f"Disponible desde {limits['risk_days']} días; hay {n}"
    monthly_bins = []
    if mvals:
        monthly_b = np.asarray([r["benchmark_ret"] for r in month_rows])
        edges = np.arange(np.floor(min(min(mvals), monthly_b.min())) - 1, np.ceil(max(max(mvals), monthly_b.max())) + 2)
        counts_a, _ = np.histogram(mvals, bins=edges)
        counts_b, _ = np.histogram(monthly_b, bins=edges)
        monthly_bins = [{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(counts_a[i]),
                         "benchmark_count": int(counts_b[i])} for i in builtins.range(len(counts_a))]
    equity_series = [{"date": d.date().isoformat(), "value": float(v), "benchmark": float(bench.loc[d]), "shadow": None} for d, v in values.items()]
    drawdown_series = [{"date": d.date().isoformat(), "dd": float(dd.loc[d]), "benchmark_dd": float(bdd.loc[d])} for d in values.index]
    rolling = []
    for window in (126, 252):
        points = []
        for i in builtins.range(window, len(returns) + 1):
            sample, bs = returns.iloc[i-window:i], breturns.iloc[i-window:i]
            vol, bvol = sample.std(ddof=1) * np.sqrt(252), bs.std(ddof=1) * np.sqrt(252)
            points.append({"date": sample.index[-1].date().isoformat(), "sharpe": float((sample.mean()-daily_rf)/sample.std(ddof=1)*np.sqrt(252)) if sample.std(ddof=1) else None,
                           "volatility": float(vol), "benchmark_sharpe": float((bs.mean()-daily_rf)/bs.std(ddof=1)*np.sqrt(252)) if bs.std(ddof=1) else None,
                           "benchmark_volatility": float(bvol)})
        rolling.append({"window_days": window, "points": points})
    troughs = dd.nsmallest(5).sort_values() if len(dd) else pd.Series(dtype=float)
    worst = [{"rank": i, "peak_date": peak_idx.date().isoformat(), "trough_date": d.date().isoformat(),
              "recovery_date": recovery_idx.date().isoformat() if recovery_idx is not None else None,
              "depth": float(value), "days_to_trough": int((d-peak_idx).days),
              "days_to_recovery": int((recovery_idx-d).days) if recovery_idx is not None else None,
              "total_days": int((recovery_idx-peak_idx).days) if recovery_idx is not None else int((end-peak_idx).days)}
             for i, (d, value) in enumerate(troughs.items(), 1)]
    sample = {"short": bool(reasons or n < limits["annualized_days"]), "reasons": sorted(reasons),
              "not_computable": [k for k, v in metrics.items() if v["status"] == "not_computable"], "unreliable": [k for k, v in metrics.items() if v["status"] == "short_sample"],
              "months_until_full": max(0, 12-complete_months), "thresholds": {k: limits[k] for k in ("descriptive_days", "risk_days", "annualized_days", "min_rebalances")}}
    report = _assemble(subject, range, align_with, start, end, n, limits, sample, metrics, benchmark_metrics, total, btotal, max_dd, max_idx,
                       recovery_idx, equity_series, drawdown_series, rolling, monthly_returns, annual_rows, monthly_bins, mvals, month_rows,
                       complete_months, daily_dist, daily_reason, worst, rf)
    if not has_bench:
        _drop_benchmark(report)
    return report


def _drop_benchmark(report: dict) -> None:
    """Sin SPY no se inventan números: todo lo relativo queda como no calculable."""
    reason = "Sin datos de SPY para este período"
    for key in report["benchmark_metrics"]:
        report["benchmark_metrics"][key] = _mv(None, "not_computable", reason)
    for key in ("alpha", "beta", "relative_return"):
        report["metrics"][key] = _mv(None, "not_computable", reason)
    sample = report["sample"]
    sample["not_computable"] = [k for k, v in report["metrics"].items() if v["status"] == "not_computable"]
    sample["unreliable"] = [k for k, v in report["metrics"].items() if v["status"] == "short_sample"]
    report["headline"]["benchmark_total_return"] = None
    for point in report["series"]["equity"]:
        point["benchmark"] = None
    for point in report["series"]["drawdown"]:
        point["benchmark_dd"] = None
    for row in report["monthly_returns"] + report["annual_returns"]:
        row["benchmark_ret"] = None
    for dist in (report["distributions"]["monthly"], report["distributions"]["daily"]):
        if dist:
            dist["benchmark_mean"] = None
            for bin_ in dist["bins"]:
                bin_["benchmark_count"] = None
    for window in report["rolling"]:
        for point in window["points"]:
            point["benchmark_sharpe"] = point["benchmark_volatility"] = None


def _assemble(subject, range, align_with, start, end, n, limits, sample, metrics, benchmark_metrics, total, btotal, max_dd, max_idx,
              recovery_idx, equity_series, drawdown_series, rolling, monthly_returns, annual_rows, monthly_bins, mvals, month_rows,
              complete_months, daily_dist, daily_reason, worst, rf):
    return {
        "subject": dict(subject), "benchmark": {"symbol": "SPY"}, "status": "ready" if n >= limits["descriptive_days"] else "insufficient_data",
        "not_computable_reason": None, "range": range, "available_ranges": [{"range": r, "available": True, "reason": None} for r in ("ytd", "1y", "3y", "5y", "all")],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat(), "trading_days": n, "complete_months": complete_months, "rebalances": None},
        "as_of": end.date().isoformat(), "stale": False, "aligned_with": align_with if isinstance(align_with, str) else None,
        "sample": sample, "flows": None, "insufficient": {"days_have": n, "days_needed": limits["descriptive_days"], "first_report_estimate": None} if n < limits["descriptive_days"] else None,
        "no_runs": None, "headline": {"total_return": total * 100, "benchmark_total_return": btotal * 100, "max_drawdown": max_dd * 100,
                    "max_dd_trough": max_idx.date().isoformat(), "max_dd_recovery_days": int((recovery_idx-max_idx).days) if recovery_idx is not None else None, "shadow_gap_pp": None},
        "metrics": metrics, "benchmark_metrics": benchmark_metrics, "series": {"equity": equity_series, "drawdown": drawdown_series},
        "rolling": rolling, "monthly_returns": monthly_returns, "annual_returns": annual_rows,
        "distributions": {"monthly": {"bin_width_pct": 1.0, "bins": monthly_bins, "mean": float(np.mean(mvals)) if mvals else 0.0,
                            "benchmark_mean": float(np.mean([r['benchmark_ret'] for r in month_rows])) if month_rows else 0.0, "n": complete_months},
                          "daily": daily_dist, "daily_reason": daily_reason}, "worst_drawdowns": worst, "cash_flows": None,
        "conventions": {"frequency": "daily", "method": "time_weighted", "annualization_days": 252, "risk_free_source": "backtest_default",
                        "risk_free_avg": rf, "benchmark": "SPY", "data_source": "backtest_export", "costs_bps": subject.get("costs_bps"),
                        "manual_url": "docs/manual/performance.md"},
    }
