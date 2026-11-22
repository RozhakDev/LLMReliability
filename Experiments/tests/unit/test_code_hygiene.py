import ast
from pathlib import Path
import re
import tokenize

import pytest

pytestmark = pytest.mark.unit

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
_LOG_PATTERN = re.compile(r"^[a-z][a-z0-9_]+(?: \| [a-z0-9_]+=[^|]+)*$")
_PLACEHOLDER_PATTERN = re.compile(r"%(?!%)[#0\- +]?(?:\d+)?(?:\.\d+)?[diouxXeEfFgGcrs]")


def test_src_package_has_no_hardcoded_paths_or_models(exp_root: Path) -> None:
    forbidden = ("/content/drive", "candidate_dirs", "davidanugraha/IndoPref", "Qwen/Qwen3-32B", "secrets.json")
    offenders = [
        (str(py_file.relative_to(exp_root)), token)
        for py_file in sorted((exp_root / "src" / "llm_exp").rglob("*.py"))
        for token in forbidden
        if token in py_file.read_text(encoding="utf-8")
    ]
    assert not offenders, f"Hardcoded tokens found: {offenders}"


def test_no_unused_imports_or_hash_comments(exp_root: Path) -> None:
    unused = []
    comments = []
    for py_file in sorted(exp_root.rglob("*.py")):
        with py_file.open("r", encoding="utf-8") as handle:
            for tok in tokenize.generate_tokens(handle.readline):
                if tok.type == tokenize.COMMENT:
                    comments.append((str(py_file.relative_to(exp_root)), tok.start[0], tok.string))

        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        imported = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported[alias.asname or alias.name.split(".")[0]] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported[alias.asname or alias.name] = node.lineno
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for name, lineno in imported.items():
            if name not in used:
                unused.append((str(py_file.relative_to(exp_root)), lineno, name))

    assert not comments, f"Hash comments found: {comments}"
    assert not unused, f"Unused imports found: {unused}"


def test_all_logger_messages_follow_structured_format_and_exact_args(exp_root: Path) -> None:
    targets = sorted(list((exp_root / "src").rglob("*.py")) + list((exp_root / "scripts").rglob("*.py")))
    errors = []
    for py_file in targets:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LOG_METHODS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                msg = node.args[0].value
                rel = str(py_file.relative_to(exp_root))
                if not _LOG_PATTERN.match(msg):
                    errors.append((rel, node.lineno, "STYLE", msg))
                expected_args = len(_PLACEHOLDER_PATTERN.findall(msg))
                actual_args = len(node.args) - 1
                if expected_args != actual_args:
                    errors.append((rel, node.lineno, "ARGC", f"{expected_args}!={actual_args}", msg))
    assert not errors, f"Invalid log messages: {errors}"
