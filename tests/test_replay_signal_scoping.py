"""Regression tests for replay signal provenance scoping."""

from datetime import datetime, timedelta

from quantagent.backtesting.backtest import Backtest, BacktestMetrics
from quantagent.models import BacktestRun, Environment, Signal, TradeSignal


def _create_backtest_run(session, name: str) -> BacktestRun:
    start = datetime(2024, 1, 1, 9, 0, 0)
    run = BacktestRun(
        name=name,
        timeframe="1h",
        assets=["BTC"],
        start_date=start,
        end_date=start + timedelta(hours=2),
        config_snapshot={},
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def _create_signal(
    session,
    *,
    run_id,
    generated_at: datetime,
    signal: TradeSignal = TradeSignal.LONG,
    environment: Environment = Environment.BACKTEST,
) -> Signal:
    db_signal = Signal(
        symbol="BTC",
        signal=signal,
        confidence=0.9,
        timeframe="1h",
        generated_at=generated_at,
        environment=environment,
        backtest_run_id=run_id,
        analysis_summary="seeded test signal",
    )
    session.add(db_signal)
    session.commit()
    session.refresh(db_signal)
    return db_signal


def _make_backtest_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        total_pnl=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        largest_win=0.0,
        largest_loss=0.0,
        total_return_pct=0.0,
    )


def _make_backtest(db_session) -> Backtest:
    start = datetime(2024, 1, 1, 9, 0, 0)
    return Backtest(
        start_date=start,
        end_date=start + timedelta(hours=2),
        assets=["BTC"],
        timeframe="1h",
        db_session=db_session,
    )


def test_create_signal_from_strategy_sets_backtest_run_id(db_session):
    run = _create_backtest_run(db_session, "source-run")
    backtest = _make_backtest(db_session)
    backtest.backtest_run_id = run.id

    created = backtest._create_signal_from_strategy(
        asset="BTC",
        decision=TradeSignal.LONG,
        confidence=0.87,
        reasoning="long breakout",
        current_date=datetime(2024, 1, 1, 10, 0, 0),
    )

    persisted = db_session.query(Signal).filter(Signal.id == created.id).one()

    assert created is not None
    assert persisted.backtest_run_id == run.id
    assert persisted.environment == Environment.BACKTEST


def test_create_signal_sets_backtest_run_id_and_thread_id(db_session):
    run = _create_backtest_run(db_session, "source-run")
    backtest = _make_backtest(db_session)
    backtest.backtest_run_id = run.id

    created = backtest._create_signal(
        asset="BTC",
        decision=TradeSignal.SHORT,
        confidence=0.63,
        result={"reasoning": "bearish continuation"},
        current_date=datetime(2024, 1, 1, 11, 0, 0),
        thread_id="backtest_123_signal_1",
    )

    persisted = db_session.query(Signal).filter(Signal.id == created.id).one()

    assert created is not None
    assert persisted.backtest_run_id == run.id
    assert persisted.thread_id == "backtest_123_signal_1"


def test_run_replay_scopes_signals_to_selected_source_run_and_records_provenance(
    db_session, monkeypatch
):
    run_a = _create_backtest_run(db_session, "run-a")
    run_b = _create_backtest_run(db_session, "run-b")
    ts1 = datetime(2024, 1, 1, 10, 0, 0)
    ts2 = datetime(2024, 1, 1, 11, 0, 0)

    sig_a1 = _create_signal(db_session, run_id=run_a.id, generated_at=ts1, signal=TradeSignal.LONG)
    sig_a2 = _create_signal(db_session, run_id=run_a.id, generated_at=ts2, signal=TradeSignal.LONG)
    sig_b1 = _create_signal(db_session, run_id=run_b.id, generated_at=ts1, signal=TradeSignal.SHORT)
    paper_signal = _create_signal(
        db_session,
        run_id=None,
        generated_at=ts1,
        signal=TradeSignal.SHORT,
        environment=Environment.PAPER,
    )

    backtest = _make_backtest(db_session)
    seen_signal_sets = []
    expected_ids = {sig_a1.id, sig_a2.id}

    monkeypatch.setattr(backtest, "_get_date_range_for_asset", lambda asset: [ts1, ts2])
    monkeypatch.setattr(
        backtest,
        "_replay_and_trade",
        lambda asset, current_date, signal_map: seen_signal_sets.append(
            {signal.id for signal in signal_map.values()}
        ),
    )
    monkeypatch.setattr(backtest, "_record_equity", lambda current_date: None)
    metrics = _make_backtest_metrics()
    monkeypatch.setattr(backtest, "_calculate_metrics", lambda: metrics)
    monkeypatch.setattr(backtest, "_update_backtest_run", lambda replay_metrics: None)

    result = backtest.run_replay(run_a.id)

    replay_run = (
        db_session.query(BacktestRun)
        .filter(BacktestRun.id == backtest.backtest_run_id)
        .one()
    )

    assert result == metrics
    assert seen_signal_sets
    assert all(signal_ids == expected_ids for signal_ids in seen_signal_sets)
    assert sig_b1.id not in seen_signal_sets[0]
    assert paper_signal.id not in seen_signal_sets[0]
    assert replay_run.replay_source_run_id == run_a.id
    assert replay_run.id != run_a.id


def test_run_replay_can_scope_to_another_overlapping_source_run(db_session, monkeypatch):
    run_a = _create_backtest_run(db_session, "run-a")
    run_b = _create_backtest_run(db_session, "run-b")
    ts1 = datetime(2024, 1, 1, 10, 0, 0)
    ts2 = datetime(2024, 1, 1, 11, 0, 0)

    _create_signal(db_session, run_id=run_a.id, generated_at=ts1, signal=TradeSignal.LONG)
    sig_b1 = _create_signal(db_session, run_id=run_b.id, generated_at=ts1, signal=TradeSignal.SHORT)
    sig_b2 = _create_signal(db_session, run_id=run_b.id, generated_at=ts2, signal=TradeSignal.SHORT)

    backtest = _make_backtest(db_session)
    seen_signal_sets = []
    expected_ids = {sig_b1.id, sig_b2.id}

    monkeypatch.setattr(backtest, "_get_date_range_for_asset", lambda asset: [ts1, ts2])
    monkeypatch.setattr(
        backtest,
        "_replay_and_trade",
        lambda asset, current_date, signal_map: seen_signal_sets.append(
            {signal.id for signal in signal_map.values()}
        ),
    )
    monkeypatch.setattr(backtest, "_record_equity", lambda current_date: None)
    monkeypatch.setattr(backtest, "_calculate_metrics", _make_backtest_metrics)
    monkeypatch.setattr(backtest, "_update_backtest_run", lambda replay_metrics: None)

    backtest.run_replay(run_b.id)

    assert seen_signal_sets
    assert all(signal_ids == expected_ids for signal_ids in seen_signal_sets)


def test_run_replay_raises_when_source_run_has_only_pre_migration_signals(db_session):
    run = _create_backtest_run(db_session, "pre-migration-run")
    ts1 = datetime(2024, 1, 1, 10, 0, 0)

    _create_signal(
        db_session,
        run_id=None,
        generated_at=ts1,
        signal=TradeSignal.LONG,
        environment=Environment.BACKTEST,
    )

    backtest = _make_backtest(db_session)

    try:
        backtest.run_replay(run.id)
        raise AssertionError("Expected ValueError for source run with NULL-scoped signals")
    except ValueError as exc:
        assert f"No stored signals found for source run {run.id}" in str(exc)
