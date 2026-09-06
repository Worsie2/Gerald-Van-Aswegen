"""``dsai doctor`` — say exactly what is wrong, and what to type to fix it.

"It doesn't work" is not a diagnosis, and neither is a stack trace. This checks
each thing that can independently break an installation, reports pass or fail
per check, and for every failure gives the one command that resolves it.

It never changes anything. Diagnosing and repairing are different jobs, and a
tool that quietly repairs is a tool that hides what was broken.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["run_checks", "render", "Check"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    #: A failure that stops the app running, as opposed to one that removes a feature.
    fatal: bool = False


#: Optional packages, what each one adds, and the extra that installs it.
_OPTIONAL = [
    ("streamlit", "the workspace UI — without it, only the command line works", "app", True),
    ("plotly", "every chart in the app and in the HTML report", "viz", True),
    ("statsmodels", "ARIMA/SARIMA, STL decomposition and OLS inference", "stats", False),
    ("xgboost", "XGBoost models", "boosting", False),
    ("lightgbm", "LightGBM models", "boosting", False),
    ("mlxtend", "Apriori and FP-Growth association mining", "mining", False),
    ("openpyxl", "reading and writing Excel files", "io", False),
    ("pyarrow", "reading and writing Parquet files", "io", False),
    ("sqlalchemy", "reading from a database", "io", False),
    ("reportlab", "writing a PDF report directly", "export", False),
    ("kaleido", "PNG chart images in Markdown, Excel and PDF exports", "export", False),
]


def _python_check() -> Check:
    version = sys.version_info
    ok = version >= (3, 10)
    return Check(
        "Python version",
        ok,
        f"{platform.python_version()} at {sys.executable}",
        "" if ok else "This platform needs Python 3.10 or newer. Install a newer Python and "
                      "recreate the virtual environment.",
        fatal=not ok,
    )


def _venv_check() -> Check:
    active = sys.prefix != sys.base_prefix
    return Check(
        "Virtual environment",
        True,
        "active" if active else "not active — packages are installing system-wide",
        "" if active else (
            "Not fatal, but recommended. From the project folder:\n"
            "      Windows:  python -m venv .venv ; .\\.venv\\Scripts\\Activate.ps1\n"
            "      macOS/Linux:  python3 -m venv .venv && source .venv/bin/activate"
        ),
    )


def _install_check() -> Check:
    try:
        import dsai

        location = Path(dsai.__file__).resolve().parent
    except Exception as exc:  # pragma: no cover - dsai is importing this module
        return Check("dsai installed", False, str(exc),
                     'Run: pip install -e ".[full]"', fatal=True)
    editable = (location.parent / "pyproject.toml").exists()
    return Check(
        "dsai installed",
        True,
        f"{location}" + (" (editable — git pull updates it immediately)" if editable
                         else " (copied into site-packages)"),
        "" if editable else (
            "Installed as a copy, so `git pull` will NOT update the running code. Reinstall "
            'from the project folder with: pip install -e ".[full]"'
        ),
    )


def _package_checks() -> list[Check]:
    checks: list[Check] = []
    for module, purpose, extra, fatal in _OPTIONAL:
        try:
            found = importlib.import_module(module)
            version = getattr(found, "__version__", "installed")
            checks.append(Check(f"  {module}", True, f"{version} — {purpose}"))
        except Exception:
            checks.append(Check(
                f"  {module}", False, f"missing — {purpose}",
                f'pip install "dsai[{extra}]"   (or: pip install {module})',
                fatal=fatal,
            ))
    return checks


def _registry_check() -> Check:
    try:
        from dsai.registry.base import REGISTRY, load_builtin_models

        load_builtin_models()
        stats = REGISTRY.stats()
        return Check(
            "Model registry",
            stats["available"] > 0,
            f"{stats['available']} of {stats['total']} algorithms available here",
            "" if stats["available"] else 'Run: pip install -e ".[full]"',
        )
    except Exception as exc:
        return Check("Model registry", False, f"{type(exc).__name__}: {exc}",
                     'Run: pip install -e ".[full]"', fatal=True)


def _config_check() -> Check:
    config = Path.cwd() / ".streamlit" / "config.toml"
    if config.exists():
        return Check("Streamlit config", True, str(config))
    return Check(
        "Streamlit config", False,
        "not found in the current folder",
        "Start the app from the project folder (the one holding pyproject.toml). Run from "
        "anywhere else and the theme and navigation settings are not picked up.",
    )


def _port_check(port: int = 8501) -> Check:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.4)
    in_use = probe.connect_ex(("127.0.0.1", port)) == 0
    probe.close()
    if not in_use:
        return Check(f"Port {port}", True, "free")
    return Check(
        f"Port {port}", False,
        "already in use — something is listening there",
        f"If that is an older copy of this app, it is serving the OLD code: stop it "
        f"(Ctrl+C in its terminal) and start again. To run alongside it instead:\n"
        f"      streamlit run dsai/app/main.py --server.port 8502",
    )


def _git_check() -> Check:
    if not shutil.which("git") or not (Path.cwd() / ".git").exists():
        return Check("Git checkout", True, "not a git checkout — updates are manual")
    try:
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:
        return Check("Git checkout", True, f"could not be read ({exc})")
    detail = f"on {branch} at {head}"
    fix = ""
    if dirty:
        count = len(dirty.splitlines())
        detail += f", {count} local change(s)"
        fix = ("Local edits will block `git pull`. Keep them with `git stash`, pull, then "
               "`git stash pop`; or discard them with `git checkout -- .`")
    return Check("Git checkout", True, detail, fix)


def _browser_check() -> Check:
    """Static chart export needs a headless browser; the HTML report does not."""
    try:
        import plotly.graph_objects as go

        go.Figure().to_image(format="png", width=10, height=10)
        return Check("Static chart export", True, "working — PNG, PDF and Excel charts will render")
    except Exception:
        return Check(
            "Static chart export", False,
            "no headless browser found",
            "Only affects PNG/PDF/Excel chart images. The HTML report's charts are interactive "
            "and need nothing extra. To enable the rest, run: plotly_get_chrome",
        )


def run_checks() -> list[Check]:
    """Every check, in the order a reader should work through them."""
    checks = [_python_check(), _venv_check(), _install_check(), _registry_check()]
    checks.append(Check("Optional packages", True, "each one adds capability; none is required"))
    checks += _package_checks()
    checks += [_config_check(), _port_check(), _git_check(), _browser_check()]
    return checks


def render(checks: list[Check]) -> str:
    lines = [
        "dsai doctor",
        f"  {platform.system()} {platform.release()} · working directory {Path.cwd()}",
        "",
    ]
    for check in checks:
        mark = "ok  " if check.ok else ("FAIL" if check.fatal else "warn")
        lines.append(f"  [{mark}] {check.name:<22} {check.detail}")
        if check.fix:
            for index, line in enumerate(check.fix.split("\n")):
                lines.append(f"           {'fix: ' if index == 0 else ''}{line}")

    broken = [c for c in checks if not c.ok and c.fatal]
    degraded = [c for c in checks if not c.ok and not c.fatal]
    lines.append("")
    if broken:
        lines.append(f"  {len(broken)} problem(s) will stop the app running. Apply the fixes above, "
                     "then run `dsai doctor` again.")
    elif degraded:
        lines.append(f"  Nothing is broken. {len(degraded)} optional package(s) are missing, each "
                     "removing a feature rather than breaking the app.")
    else:
        lines.append("  Everything checks out. Start the workspace with:  dsai app")
    return "\n".join(lines)
