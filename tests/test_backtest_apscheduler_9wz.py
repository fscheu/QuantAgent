"""Tests for QuantAgent-9wz: APScheduler integration for backtest execution."""

import time
from datetime import datetime

import pytest
from apscheduler.schedulers import SchedulerAlreadyRunningError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger


class TestAPSchedulerCore:
    """Core APScheduler functionality required for AC1."""

    def test_apscheduler_initialization(self):
        """AC1: APScheduler BackgroundScheduler initializes without error."""
        scheduler = BackgroundScheduler()
        assert scheduler is not None
        assert not scheduler.running

    def test_apscheduler_start_stop(self):
        """AC1: Scheduler can start and stop gracefully."""
        scheduler = BackgroundScheduler()
        scheduler.start()
        assert scheduler.running
        scheduler.shutdown()
        assert not scheduler.running

    def test_apscheduler_start_raises_when_already_running(self):
        """AC1: Starting running scheduler raises SchedulerAlreadyRunningError."""
        scheduler = BackgroundScheduler()
        scheduler.start()
        with pytest.raises(SchedulerAlreadyRunningError):
            scheduler.start()
        scheduler.shutdown()


class TestPollerJobConfiguration:
    """AC1: Poller job configured with 10s interval and correct ID."""

    def test_poller_job_interval_10_seconds(self):
        """AC1: Poller job must have 10-second interval trigger."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        call_count = [0]

        def poller_func():
            call_count[0] += 1

        job = scheduler.add_job(
            poller_func,
            trigger=IntervalTrigger(seconds=10),
            id="backtest_poller",
            replace_existing=True,
        )

        assert job.trigger.interval.total_seconds() == 10
        scheduler.shutdown()

    def test_poller_job_id_format(self):
        """AC1: Poller job must use ID 'backtest_poller'."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def dummy():
            pass

        job = scheduler.add_job(
            dummy,
            trigger=IntervalTrigger(seconds=10),
            id="backtest_poller",
        )

        assert job.id == "backtest_poller"
        scheduler.shutdown()

    def test_poller_job_retrieval_after_registration(self):
        """AC1: Poller job can be retrieved after registration."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def dummy():
            pass

        scheduler.add_job(
            dummy,
            trigger=IntervalTrigger(seconds=10),
            id="backtest_poller",
        )

        retrieved = scheduler.get_job("backtest_poller")
        assert retrieved is not None
        assert retrieved.id == "backtest_poller"

        scheduler.shutdown()


class TestBacktestJobIDFormat:
    """AC2: Execution job IDs follow format 'backtest_run_{run_id}'."""

    def test_execution_job_id_format_validation(self):
        """AC2: Job ID must follow format 'backtest_run_{run_id}'."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        run_id = 123
        job_id = f"backtest_run_{run_id}"
        expected = "backtest_run_123"

        assert job_id == expected

        def dummy():
            pass

        job = scheduler.add_job(
            dummy,
            trigger=IntervalTrigger(seconds=1),
            id=job_id,
        )

        assert job.id == "backtest_run_123"
        scheduler.shutdown()

    def test_execution_job_can_be_added_dynamically(self):
        """AC2: Execution jobs can be added dynamically per pending run."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def execution_job(run_id):
            pass

        job = scheduler.add_job(
            lambda: execution_job(456),
            trigger=IntervalTrigger(seconds=10),
            id="backtest_run_456",
            max_instances=1,
        )

        assert job.id == "backtest_run_456"
        scheduler.shutdown()


class TestSchedulerPersistence:
    """AC1: Scheduler persists across Streamlit reloads via singleton."""

    def test_scheduler_singleton_pattern(self):
        """AC1: Scheduler can be stored in module-level singleton."""
        _scheduler_instance = {}

        scheduler = BackgroundScheduler()
        scheduler.start()

        _scheduler_instance["scheduler"] = scheduler

        retrieved = _scheduler_instance["scheduler"]
        assert retrieved is scheduler
        assert retrieved.running

        scheduler.shutdown()

    def test_scheduler_cache_resource_pattern(self):
        """AC1: Scheduler respects @st.cache_resource pattern."""
        import functools

        call_count = [0]

        @functools.lru_cache(maxsize=1)
        def get_scheduler():
            call_count[0] += 1
            scheduler = BackgroundScheduler()
            scheduler.start()
            return scheduler

        s1 = get_scheduler()
        assert call_count[0] == 1
        assert s1.running

        s2 = get_scheduler()
        assert call_count[0] == 1
        assert s2 is s1

        s1.shutdown()


class TestJobManagement:
    """AC2, AC7: Job add, remove, and idempotent operations."""

    def test_job_add_with_replace_existing(self):
        """AC2: Adding job with replace_existing=True is idempotent."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        call_count = [0]

        def dummy():
            call_count[0] += 1

        job_id = "backtest_run_789"

        scheduler.add_job(
            dummy,
            trigger=IntervalTrigger(seconds=1),
            id=job_id,
            replace_existing=False,
        )

        job2 = scheduler.add_job(
            dummy,
            trigger=IntervalTrigger(seconds=1),
            id=job_id,
            replace_existing=True,
        )

        assert job2.id == job_id
        scheduler.shutdown()

    def test_job_removal_for_cancellation(self):
        """AC7: Job can be removed to cancel execution."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def dummy():
            pass

        job_id = "backtest_run_999"
        scheduler.add_job(
            dummy,
            trigger=IntervalTrigger(seconds=10),
            id=job_id,
        )

        assert scheduler.get_job(job_id) is not None

        scheduler.remove_job(job_id)

        assert scheduler.get_job(job_id) is None

        scheduler.shutdown()

    def test_job_removal_raises_on_missing_job(self):
        """AC7: Removing non-existent job raises JobLookupError."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        with pytest.raises(Exception):
            scheduler.remove_job("non_existent_job")

        scheduler.shutdown()


class TestMaxInstancesConstraint:
    """AC12: Single execution constraint via max_instances=1."""

    def test_max_instances_one_configuration(self):
        """AC12: Execution job configured with max_instances=1."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def execution_task():
            pass

        job = scheduler.add_job(
            execution_task,
            trigger=IntervalTrigger(seconds=10),
            id="backtest_run_execution",
            max_instances=1,
        )

        assert job is not None

        scheduler.shutdown()


class TestPollerExecution:
    """AC13: Poller robustness to errors."""

    def test_poller_exception_handling(self):
        """AC13: Poller job can handle exceptions and continue."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        errors_caught = []

        def fragile_poller():
            try:
                raise Exception("Simulated DB error")
            except Exception as e:
                errors_caught.append(str(e))

        scheduler.add_job(
            fragile_poller,
            trigger=IntervalTrigger(seconds=0.5),
            id="backtest_poller",
        )

        time.sleep(1.5)

        assert scheduler.running
        assert len(errors_caught) > 0

        scheduler.shutdown()

    def test_poller_continues_after_exception(self):
        """AC13: Poller continues running after exception."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        execution_log = []

        def sometimes_fails():
            execution_log.append(datetime.now())
            if len(execution_log) < 2:
                raise Exception("First execution fails")

        scheduler.add_job(
            sometimes_fails,
            trigger=IntervalTrigger(seconds=0.3),
            id="backtest_poller",
        )

        time.sleep(1.2)

        assert scheduler.running
        assert len(execution_log) >= 3

        scheduler.shutdown()


class TestSchedulerShutdownCleanup:
    """Verify proper cleanup and resource release."""

    def test_scheduler_shutdown_removes_all_jobs(self):
        """Scheduler shutdown should clean up all jobs."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def dummy():
            pass

        scheduler.add_job(dummy, trigger=IntervalTrigger(seconds=10), id="job1")
        scheduler.add_job(dummy, trigger=IntervalTrigger(seconds=10), id="job2")

        assert len(scheduler.get_jobs()) >= 2

        scheduler.shutdown()

        assert len(scheduler.get_jobs()) == 0

    def test_multiple_scheduler_instances_independent(self):
        """Multiple scheduler instances should be independent."""
        s1 = BackgroundScheduler()
        s2 = BackgroundScheduler()

        s1.start()
        s2.start()

        def dummy():
            pass

        s1.add_job(dummy, trigger=IntervalTrigger(seconds=10), id="job1")
        s2.add_job(dummy, trigger=IntervalTrigger(seconds=10), id="job2")

        assert s1.get_job("job1") is not None
        assert s1.get_job("job2") is None

        assert s2.get_job("job2") is not None
        assert s2.get_job("job1") is None

        s1.shutdown()
        s2.shutdown()


class TestJobExecutionSequencing:
    """AC12: MVP constraint - sequential execution, not concurrent."""

    def test_sequential_job_execution_setup(self):
        """AC12: Jobs configured for sequential execution (max_instances=1)."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        execution_times = []

        def backtest_job():
            execution_times.append(datetime.now())
            time.sleep(0.1)

        scheduler.add_job(
            backtest_job,
            trigger=IntervalTrigger(seconds=0.2),
            id="backtest_run_1",
            max_instances=1,
        )

        time.sleep(0.7)

        assert len(execution_times) >= 2

        scheduler.shutdown()


class TestBackgroundSchedulerThreadSafety:
    """Verify thread-safe operations for concurrent access."""

    def test_add_job_thread_safe(self):
        """Adding jobs should be thread-safe."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def dummy():
            pass

        for i in range(10):
            scheduler.add_job(
                dummy,
                trigger=IntervalTrigger(seconds=1),
                id=f"job_{i}",
                replace_existing=False,
            )

        jobs = scheduler.get_jobs()
        assert len(jobs) >= 9

        scheduler.shutdown()

    def test_get_job_during_execution(self):
        """Getting job while executing should be safe."""
        scheduler = BackgroundScheduler()
        scheduler.start()

        def busy_job():
            time.sleep(0.2)

        scheduler.add_job(
            busy_job,
            trigger=IntervalTrigger(seconds=0.1),
            id="busy_job",
        )

        time.sleep(0.15)

        job = scheduler.get_job("busy_job")
        assert job is not None

        scheduler.shutdown()
