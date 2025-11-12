import subprocess
import sys
from pathlib import Path


from clockify_support_cli_final import run_selftest


def test_run_selftest_returns_true():
    assert run_selftest() is True


def test_cli_selftest_exit_zero():
    script = Path(__file__).resolve().parents[1] / "clockify_support_cli_final.py"
    result = subprocess.run(
        [sys.executable, str(script), "--selftest"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Selftest exited with {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
