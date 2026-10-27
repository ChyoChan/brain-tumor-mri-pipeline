#!/usr/bin/env python3
"""Stop only this project's local MRI UI (default http://127.0.0.1:8765)."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

try:
    from .run_frontend import DEFAULT_PORT, port_number
except ImportError:  # Direct invocation: python scripts/stop_frontend.py
    from run_frontend import DEFAULT_PORT, port_number

ROOT = Path(__file__).resolve().parents[1]


def listening_pids(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pid = int(line)
            if pid != os.getpid():
                pids.append(pid)
    return sorted(set(pids))


def is_frontend_process(pid: int) -> bool:
    """Do not signal another project's server or an unrelated port owner."""
    try:
        command = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        cwd_output = subprocess.check_output(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    is_runner = "run_frontend.py" in command or (
        "uvicorn" in command and "app.server:app" in command
    )
    return is_runner and f"n{ROOT}" in cwd_output.splitlines()


def stop(pids: list[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
            print(f"sent {signal.Signals(sig).name} to pid {pid}")
        except ProcessLookupError:
            print(f"pid {pid} already gone")
        except PermissionError as exc:
            print(f"could not signal pid {pid}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    listeners = listening_pids(args.port)
    if not listeners:
        print(f"nothing listening on 127.0.0.1:{args.port}")
        return 0
    pids = [pid for pid in listeners if is_frontend_process(pid)]
    if len(pids) != len(listeners):
        print(f"port {args.port} has an unrelated or unverified process; refusing to stop it")
        return 1

    stop(pids, signal.SIGTERM)
    deadline = time.monotonic() + 5
    leftover = pids
    while leftover and time.monotonic() < deadline:
        time.sleep(0.2)
        leftover = [pid for pid in listening_pids(args.port) if pid in pids]
    if leftover:
        # Recheck identity before escalation in case a process exited meanwhile.
        leftover = [pid for pid in leftover if is_frontend_process(pid)]
        stop(leftover, signal.SIGKILL)
        time.sleep(0.2)

    if listening_pids(args.port):
        print(f"port {args.port} is still in use")
        return 1
    print(f"port {args.port} is free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
