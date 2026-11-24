import ast
from pathlib import Path
import re
import subprocess
import sys
import tokenize

import pytest

pytestmark = pytest.mark.unit

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
_LOG_PATTERN = re.compile(r"^[a-z][a-z0-9_]+(?: \| [a-z0-9_]+=[^|]+)*$")
_PLACEHOLDER_PATTERN = re.compile(r"%(?!%)[#0\- +]?(?:\d+)?(?:\.\d+)?[diouxXeEfFgGcrs]")


def test_src_package_has_no_hardcoded_paths_or_models(project_dir: Path) -> None:
    forbidden = (
        "/content/drive",
        "davidanugraha/IndoPref",
        "Qwen/Qwen3-32B",
        "secrets.json",
        "mistral",
        "gemma3",
        "aya-expanse",
        "GLM-4",
    )
    offenders = [
        (str(py_file.relative_to(project_dir)), token)
        for py_file in sorted((project_dir / "src" / "qwen3_eval").rglob("*.py"))
        for token in forbidden
        if token.lower() in py_file.read_text(encoding="utf-8").lower()
    ]
    assert not offenders, f"Hardcoded or non-Qwen3 tokens found: {offenders}"


def test_no_unused_imports_docstrings_or_hash_comments(project_dir: Path) -> None:
    unused = []
    comments = []
    docstrings = []
    for py_file in sorted(project_dir.rglob("*.py")):
        rel = str(py_file.relative_to(project_dir))
        with py_file.open("r", encoding="utf-8") as handle:
            for tok in tokenize.generate_tokens(handle.readline):
                if tok.type == tokenize.COMMENT:
                    comments.append((rel, tok.start[0], tok.string))

        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    docstrings.append((rel, node.body[0].lineno))

        if py_file.name == "__init__.py":
            continue
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
                unused.append((rel, lineno, name))

    assert not comments, f"Hash comments found: {comments}"
    assert not docstrings, f"Docstrings found: {docstrings}"
    assert not unused, f"Unused imports found: {unused}"


def test_all_logger_messages_follow_structured_format_and_exact_args(project_dir: Path) -> None:
    targets = sorted(list((project_dir / "src").rglob("*.py")) + list((project_dir / "scripts").rglob("*.py")))
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
                rel = str(py_file.relative_to(project_dir))
                if not _LOG_PATTERN.match(msg):
                    errors.append((rel, node.lineno, "STYLE", msg))
                expected_args = len(_PLACEHOLDER_PATTERN.findall(msg))
                actual_args = len(node.args) - 1
                if expected_args != actual_args:
                    errors.append((rel, node.lineno, "ARGC", f"{expected_args}!={actual_args}", msg))
    assert not errors, f"Invalid log messages: {errors}"


def test_config_and_logger_imports_do_not_eagerly_load_heavy_ml_libs(project_dir: Path) -> None:
    src_dir = str(project_dir / "src")
    code = (
        f"import sys; sys.path.insert(0, {src_dir!r}); "
        "import qwen3_eval.config; import qwen3_eval.utils.logger; "
        "heavy = [m for m in ('torch', 'transformers', 'statsmodels') if m in sys.modules]; "
        "assert not heavy, f'Unexpected heavy imports: {heavy}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
