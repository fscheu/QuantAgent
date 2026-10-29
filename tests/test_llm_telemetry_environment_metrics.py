from __future__ import annotations

from datetime import datetime, timedelta

from quantagent.llm_telemetry import get_environment_metrics
from quantagent.models import Log


def _log(db, environment="paper", event_type="llm_call", input_tokens=10,
         output_tokens=5, total_tokens=15, duration_ms=100.0, minutes_ago=5):
    row = Log(
        timestamp=datetime.utcnow() - timedelta(minutes=minutes_ago),
        level="INFO",
        module="quantagent.llm_telemetry",
        message=f"llm_call test {environment}",
        event_type=event_type,
        environment=environment,
        extra_data={
            "operation": "test",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
        },
    )
    db.add(row)
    db.commit()
    return row

class TestGetEnvironmentMetrics:
    """Unit tests for llm_telemetry.get_environment_metrics."""
    def test_empty_returns_zero_aggregate(self, db_session):
        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 0
        assert result["input_tokens_sum"] == 0
        assert result["output_tokens_sum"] == 0
        assert result["total_tokens_sum"] == 0
        assert result["duration_ms_sum"] == 0.0
        assert result["duration_ms_avg"] == 0.0
        assert result["by_operation"] == {}

    def test_returns_required_keys(self, db_session):
        result = get_environment_metrics(db_session, environment="paper")
        required = {
            "calls", "input_tokens_sum", "output_tokens_sum", "total_tokens_sum",
            "duration_ms_sum", "duration_ms_avg", "by_operation",
        }
        assert required.issubset(result.keys())

    def test_filters_by_environment_paper(self, db_session):
        _log(db_session, environment="paper", input_tokens=100)
        _log(db_session, environment="paper", input_tokens=200)
        _log(db_session, environment="backtest", input_tokens=9999)
        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 2
        assert result["input_tokens_sum"] == 300

    def test_filters_by_environment_backtest(self, db_session):
        _log(db_session, environment="paper", input_tokens=9999)
        _log(db_session, environment="backtest", input_tokens=50)
        result = get_environment_metrics(db_session, environment="backtest")
        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 50

    def test_excludes_rows_outside_time_window(self, db_session):
        _log(db_session, environment="paper", input_tokens=10, minutes_ago=30)
        _log(db_session, environment="paper", input_tokens=9999, minutes_ago=49 * 60)
        result = get_environment_metrics(db_session, environment="paper", hours_back=48)
        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 10

    def test_excludes_non_llm_call_events(self, db_session):
        _log(db_session, environment="paper", event_type="heartbeat", input_tokens=9999)
        _log(db_session, environment="paper", event_type="llm_call", input_tokens=10)
        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 1
        assert result["input_tokens_sum"] == 10

    def test_handles_null_tokens_gracefully(self, db_session):
        _log(db_session, environment="paper", input_tokens=None, output_tokens=None, total_tokens=None)
        _log(db_session, environment="paper", input_tokens=20, output_tokens=10, total_tokens=30)
        result = get_environment_metrics(db_session, environment="paper")
        assert result["calls"] == 2
        assert result["input_tokens_sum"] == 20
        assert result["output_tokens_sum"] == 10

    def test_aggregates_duration_and_avg(self, db_session):
        _log(db_session, environment="paper", duration_ms=100.0)
        _log(db_session, environment="paper", duration_ms=300.0)
        result = get_environment_metrics(db_session, environment="paper")
        assert result["duration_ms_sum"] == 400.0
        assert result["duration_ms_avg"] == 200.0

    def test_by_operation_breakdown_present(self, db_session):
        _log(db_session, environment="paper", input_tokens=100)
        _log(db_session, environment="paper", input_tokens=200)
        result = get_environment_metrics(db_session, environment="paper")
        assert "by_operation" in result
        assert "test" in result["by_operation"]
        assert result["by_operation"]["test"]["calls"] == 2
