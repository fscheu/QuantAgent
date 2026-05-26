from __future__ import annotations

from apps.streamlit.views import paper_trading


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self, *, start_clicked: bool = False, stop_clicked: bool = False):
        self.start_clicked = start_clicked
        self.stop_clicked = stop_clicked
        self.button_calls: list[dict] = []
        self.warning_messages: list[str] = []
        self.rerun_called = False

    def subheader(self, *_args, **_kwargs):
        return None

    def expander(self, *_args, **_kwargs):
        return _DummyContext()

    def text_input(self, *_args, **_kwargs):
        return "BTC,SPX"

    def radio(self, *_args, **_kwargs):
        return "Single cycle"

    def number_input(self, *_args, **_kwargs):
        return 1.0

    def warning(self, message: str):
        self.warning_messages.append(message)

    def button(self, label: str, **kwargs):
        self.button_calls.append({"label": label, **kwargs})
        if kwargs.get("key") == "sc_start":
            return self.start_clicked
        if kwargs.get("key") == "sc_stop":
            return self.stop_clicked
        return False

    def columns(self, spec):
        return [_DummyContext() for _ in spec]

    def rerun(self):
        self.rerun_called = True


class _FakeDB:
    def __init__(self, heartbeat):
        self._heartbeat = heartbeat

    def get_latest_heartbeat(self, _environment: str):
        return self._heartbeat


def test_read_pid_returns_none_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_trading, "_PID_FILE", tmp_path / "missing.pid")
    assert paper_trading._read_pid() is None


def test_read_pid_returns_int_when_valid(tmp_path, monkeypatch):
    pid_file = tmp_path / "scheduler.pid"
    pid_file.write_text("12345\n")
    monkeypatch.setattr(paper_trading, "_PID_FILE", pid_file)
    assert paper_trading._read_pid() == 12345


def test_read_pid_returns_none_on_invalid_content(tmp_path, monkeypatch):
    pid_file = tmp_path / "scheduler.pid"
    pid_file.write_text("not-a-pid\n")
    monkeypatch.setattr(paper_trading, "_PID_FILE", pid_file)
    assert paper_trading._read_pid() is None


def test_pid_is_alive_handles_none():
    assert paper_trading._pid_is_alive(None) is False


def test_write_and_clear_pid(tmp_path, monkeypatch):
    pid_file = tmp_path / "scheduler.pid"
    monkeypatch.setattr(paper_trading, "_PID_FILE", pid_file)

    paper_trading._write_pid(12345)
    assert pid_file.read_text() == "12345"

    paper_trading._clear_pid()
    assert not pid_file.exists()


def test_launch_subprocess_single_cycle_writes_pid(monkeypatch, tmp_path):
    pid_file = tmp_path / "scheduler.pid"
    repo_root = tmp_path / "repo"
    (repo_root / "apps").mkdir(parents=True)
    popen_calls = []

    class FakeProc:
        pid = 99999

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(paper_trading, "_PID_FILE", pid_file)
    monkeypatch.setattr(paper_trading, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(paper_trading.subprocess, "Popen", fake_popen)

    paper_trading._launch_subprocess("BTC,SPX", "Single cycle", 1.0, "paper")

    assert pid_file.read_text() == "99999"
    cmd, kwargs = popen_calls[0]
    assert cmd[1].endswith("apps/paper_trading.py")
    assert cmd[2:7] == ["--environment", "paper", "--assets", "BTC,SPX", "--enable"]
    assert "--run-once" in cmd
    assert kwargs["cwd"] == str(repo_root)
    assert kwargs["start_new_session"] is True


def test_launch_subprocess_continuous_mode_uses_interval(monkeypatch, tmp_path):
    pid_file = tmp_path / "scheduler.pid"
    repo_root = tmp_path / "repo"
    (repo_root / "apps").mkdir(parents=True)
    popen_calls = []

    class FakeProc:
        pid = 777

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(paper_trading, "_PID_FILE", pid_file)
    monkeypatch.setattr(paper_trading, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(paper_trading.subprocess, "Popen", fake_popen)

    paper_trading._launch_subprocess("BTC,SPX", "Continuous", 0.5, "paper")

    cmd, _kwargs = popen_calls[0]
    assert "--run-once" not in cmd
    assert cmd[-2:] == ["--interval-hours", "0.5"]


def test_stop_subprocess_sends_term_then_kill_and_clears_pid(monkeypatch, tmp_path):
    pid_file = tmp_path / "scheduler.pid"
    pid_file.write_text("12345")
    monkeypatch.setattr(paper_trading, "_PID_FILE", pid_file)
    monkeypatch.setattr(paper_trading.time, "sleep", lambda *_args, **_kwargs: None)

    kill_calls = []

    def fake_kill(pid, sig):
        kill_calls.append((pid, sig))

    liveness = iter([True])
    monkeypatch.setattr(paper_trading.os, "kill", fake_kill)
    monkeypatch.setattr(paper_trading, "_pid_is_alive", lambda _pid: next(liveness, False))

    paper_trading._stop_subprocess(12345)

    assert kill_calls == [(12345, paper_trading.signal.SIGTERM), (12345, paper_trading.signal.SIGKILL)]
    assert not pid_file.exists()


def test_render_scheduler_controls_disables_start_when_running(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(paper_trading, "st", fake_st)
    monkeypatch.setattr(paper_trading, "_read_pid", lambda: 4242)
    monkeypatch.setattr(paper_trading, "_pid_is_alive", lambda _pid: True)

    db = _FakeDB({"status": "running"})
    paper_trading._render_scheduler_controls(db, "paper")

    start_call = next(call for call in fake_st.button_calls if call["key"] == "sc_start")
    stop_call = next(call for call in fake_st.button_calls if call["key"] == "sc_stop")
    assert start_call["disabled"] is True
    assert stop_call["disabled"] is False
    assert any("already running" in msg for msg in fake_st.warning_messages)


def test_render_scheduler_controls_disables_stop_when_stopped(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(paper_trading, "st", fake_st)
    monkeypatch.setattr(paper_trading, "_read_pid", lambda: None)
    monkeypatch.setattr(paper_trading, "_pid_is_alive", lambda _pid: False)

    db = _FakeDB(None)
    paper_trading._render_scheduler_controls(db, "paper")

    start_call = next(call for call in fake_st.button_calls if call["key"] == "sc_start")
    stop_call = next(call for call in fake_st.button_calls if call["key"] == "sc_stop")
    assert start_call["disabled"] is False
    assert stop_call["disabled"] is True


def test_render_scheduler_controls_start_button_launches(monkeypatch):
    fake_st = _FakeStreamlit(start_clicked=True)
    launch_calls = []
    monkeypatch.setattr(paper_trading, "st", fake_st)
    monkeypatch.setattr(paper_trading, "_read_pid", lambda: None)
    monkeypatch.setattr(paper_trading, "_pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(paper_trading.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        paper_trading,
        "_launch_subprocess",
        lambda assets, mode, interval_hours, environment: launch_calls.append(
            (assets, mode, interval_hours, environment)
        ),
    )

    db = _FakeDB(None)
    paper_trading._render_scheduler_controls(db, "paper")

    assert launch_calls == [("BTC,SPX", "Single cycle", 1.0, "paper")]
    assert fake_st.rerun_called is True


def test_render_scheduler_controls_stop_button_stops(monkeypatch):
    fake_st = _FakeStreamlit(stop_clicked=True)
    stop_calls = []
    monkeypatch.setattr(paper_trading, "st", fake_st)
    monkeypatch.setattr(paper_trading, "_read_pid", lambda: 5150)
    monkeypatch.setattr(paper_trading, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(paper_trading.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(paper_trading, "_stop_subprocess", lambda pid: stop_calls.append(pid))

    db = _FakeDB({"status": "running"})
    paper_trading._render_scheduler_controls(db, "paper")

    assert stop_calls == [5150]
    assert fake_st.rerun_called is True
