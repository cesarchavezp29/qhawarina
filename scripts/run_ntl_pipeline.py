"""Launcher for update_ntl_monthly.py — sets UTF-8, writes log, runs pipeline."""
import subprocess, sys
from pathlib import Path

log_path = Path(r"D:\nexus\nexus\data\raw\satellite\ntl_pipeline_run.log")
script   = Path(r"D:\nexus\nexus\scripts\update_ntl_monthly.py")

args = [
    sys.executable, str(script),
    "--start-date", "2025-02",
    "--no-force-clean",
    "--skip-rossi",
]

print(f"Launching NTL pipeline. Log: {log_path}")
with open(log_path, "w", encoding="utf-8") as log:
    proc = subprocess.Popen(
        args,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=r"D:\nexus\nexus",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    print(f"PID: {proc.pid}")
    proc.wait()
    print(f"Exit code: {proc.returncode}")
