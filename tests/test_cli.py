"""dsai/cli.py, invoked the way a user actually invokes it.

Every other test reaches ``main()`` by importing it, which never executes the
module's own ``if __name__ == "__main__":`` guard. ``python -m dsai.cli`` does
execute it -- and does so top-to-bottom, so a handler function defined after
that guard is not yet bound when ``main()`` runs from inside it. That exact
ordering bug (the guard sat before ``_doctor``/``_version``/``_predict`` were
defined) shipped unnoticed because nothing here ran the file as a script.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dsai.cli", *args],
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("command", ["doctor", "version", "--help"])
def test_module_execution_does_not_crash(command):
    """Every subcommand must be reachable when run as `python -m dsai.cli ...`,
    not only when `main` is imported and called directly."""
    result = _run(command)
    assert result.returncode in (0, 1), (
        f"`python -m dsai.cli {command}` crashed:\n{result.stderr}"
    )
    assert "NameError" not in result.stderr
    assert "Traceback" not in result.stderr


def test_every_registered_subcommand_has_a_bound_handler():
    """Every name accepted by argparse must resolve to an already-defined
    function by the time the handler dict is built -- catches the same bug
    for any future subcommand, not just the three it hit this time."""
    import ast
    from pathlib import Path

    import dsai.cli

    source = Path(dsai.cli.__file__).read_text()
    tree = ast.parse(source)

    guard_line = next(
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
    )
    function_lines = {
        node.name: node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    handler_names = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
        and node.name not in {"_load"}
    ]
    late = {name: function_lines[name] for name in handler_names
            if function_lines[name] > guard_line}
    assert not late, (
        f"these handlers are defined after the `if __name__ == '__main__':` "
        f"guard at line {guard_line}, so `python -m dsai.cli` calls main() "
        f"before they exist: {late}"
    )
