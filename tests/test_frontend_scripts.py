import argparse
import os
import signal
import subprocess
from pathlib import Path

import pytest
import uvicorn

from scripts import run_frontend, stop_frontend


def test_runner_uses_new_default_port(monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    run_frontend.main([])
    assert calls[0][0] == ("app.server:app",)
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 8765


def test_runner_accepts_custom_port(monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    run_frontend.main(["--port", "8766"])
    assert calls[0]["port"] == 8766


@pytest.mark.parametrize("value", ["0", "-1", "65536", "not-a-port"])
def test_invalid_port_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        run_frontend.port_number(value)


def test_stopper_defaults_to_new_port(monkeypatch):
    ports = []
    monkeypatch.setattr(stop_frontend, "listening_pids", lambda port: ports.append(port) or [])
    assert stop_frontend.main([]) == 0
    assert ports == [8765]


def test_stopper_refuses_unrelated_listener(monkeypatch):
    monkeypatch.setattr(stop_frontend, "listening_pids", lambda port: [123])
    monkeypatch.setattr(stop_frontend, "is_frontend_process", lambda pid: False)
    monkeypatch.setattr(stop_frontend, "stop", lambda *args: pytest.fail("must not signal unrelated process"))
    assert stop_frontend.main([]) == 1


def test_stopper_signals_only_verified_listener_on_selected_port(monkeypatch):
    polls = iter([[123], [], []])
    ports, signals = [], []

    def listeners(port):
        ports.append(port)
        return next(polls)

    monkeypatch.setattr(stop_frontend, "listening_pids", listeners)
    monkeypatch.setattr(stop_frontend, "is_frontend_process", lambda pid: pid == 123)
    monkeypatch.setattr(stop_frontend, "stop", lambda pids, sig: signals.append((pids, sig)))
    monkeypatch.setattr(stop_frontend.time, "sleep", lambda seconds: None)
    assert stop_frontend.main(["--port", "8766"]) == 0
    assert ports == [8766, 8766, 8766]
    assert signals == [([123], signal.SIGTERM)]


@pytest.mark.parametrize(
    "command,cwd,is_ours",
    [
        ("python scripts/run_frontend.py", None, True),
        ("uvicorn app.server:app", None, True),
        ("python scripts/run_frontend.py", "/another/project", False),
        ("python other_server.py", None, False),
    ],
)
def test_process_identity_requires_project_and_frontend(monkeypatch, command, cwd, is_ours):
    replies = iter([command, f"p123\nfcwd\nn{cwd or stop_frontend.ROOT}\n"])
    monkeypatch.setattr(stop_frontend.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    assert stop_frontend.is_frontend_process(123) is is_ours


@pytest.mark.parametrize(
    "launcher,entrypoint",
    [("start_frontend.command", "run_frontend.py"), ("stop_frontend.command", "stop_frontend.py")],
)
def test_launchers_work_from_other_directories_with_quoted_paths(tmp_path, launcher, entrypoint):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    # Verify working-directory and argument handling without starting a server.
    fake_conda = tmp_path / "fake conda"
    fake_conda.write_text(
        "#!/bin/bash\n"
        'printf "cwd=%s\\n" "$PWD"\n'
        'printf "arg=%s\\n" "$@"\n'
        "exit 0\n"
    )
    fake_conda.chmod(0o755)
    env = dict(os.environ, MRI_CONDA_EXE=str(fake_conda), MRI_CONDA_ENV="test env")
    result = subprocess.run(
        [str(scripts / launcher), "--port", "8766"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    lines = result.stdout.splitlines()
    assert f"cwd={scripts.parent}" in lines
    assert [line.removeprefix("arg=") for line in lines if line.startswith("arg=")] == [
        "run", "--no-capture-output", "-n", "test env", "env", "PYTHONUNBUFFERED=1",
        "python", str(scripts / entrypoint), "--port", "8766",
    ]


@pytest.mark.parametrize("launcher", ["start_frontend.command", "stop_frontend.command"])
def test_launchers_report_invalid_conda_path(tmp_path, launcher):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    env = dict(os.environ, MRI_CONDA_EXE=str(tmp_path / "not-installed"))
    result = subprocess.run(
        [str(scripts / launcher)], cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Conda not found" in result.stdout
