#!/usr/bin/env python3
"""Stop the local MRI pipeline UI (default http://127.0.0.1:8000)."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


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


def related_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "scripts/run_frontend.py|uvicorn .*app.server:app"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        if line.strip().isdigit():
            pid = int(line.strip())
            if pid != os.getpid():
                pids.append(pid)
    return sorted(set(pids))


def stop(pids: list[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
            print(f"sent {signal.Signals(sig).name} to pid {pid}")
        except ProcessLookupError:
            print(f"pid {pid} already gone")
        except PermissionError as exc:
            print(f"could not signal pid {pid}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    pids = listening_pids(args.port)
    pids = sorted(set(pids + related_pids()))
    if not pids:
        print(f"nothing listening on 127.0.0.1:{args.port}")
        return 0

    stop(pids, signal.SIGTERM)
    time.sleep(0.4)
    leftover = [pid for pid in listening_pids(args.port) if pid in pids]
    leftover += [pid for pid in related_pids() if pid in pids]
    leftover = sorted(set(leftover))
    if leftover:
        stop(leftover, signal.SIGKILL)
        time.sleep(0.2)

    if listening_pids(args.port):
        print(f"port {args.port} is still in use")
        return 1
    print(f"port {args.port} is free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
