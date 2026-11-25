import ast
import io
import json
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
_BANNER_PATTERN = re.compile(r"(-{3,}|={3,}|\*{4,}|['\"][-=\*]['\"]\s*\*\s*\d+|[\u2500-\u257f])")
_EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"


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


def test_no_print_banners_comments_or_slop_in_code_and_notebook(project_dir: Path) -> None:
    issues = []
    py_targets = sorted(
        list((project_dir / "src").rglob("*.py"))
        + list((project_dir / "scripts").rglob("*.py"))
        + list((project_dir / "tests").rglob("*.py"))
    )
    for py_file in py_targets:
        rel = py_file.relative_to(project_dir).as_posix()
        text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                issues.append((rel, node.lineno, "print"))
        for lineno, line in enumerate(text.splitlines(), 1):
            if _BANNER_PATTERN.search(line):
                issues.append((rel, lineno, "banner"))
            if _EMOJI_PATTERN.search(line):
                issues.append((rel, lineno, "emoji"))
            if _EM_DASH in line or _EN_DASH in line:
                issues.append((rel, lineno, "dash"))

    nb_path = project_dir / "01_Qwen3_Evaluation_Runner.ipynb"
    if nb_path.is_file():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            cid = cell["id"]
            src = "".join(cell["source"])
            if cell["cell_type"] == "code":
                cleaned = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!"))
                for tok in tokenize.generate_tokens(io.StringIO(cleaned).readline):
                    if tok.type == tokenize.COMMENT:
                        issues.append((f"nb:{cid}", tok.start[0], "comment"))
                tree = ast.parse(cleaned)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                        issues.append((f"nb:{cid}", node.lineno, "print"))
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        if (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        ):
                            issues.append((f"nb:{cid}", node.body[0].lineno, "docstring"))
            for lineno, line in enumerate(src.splitlines(), 1):
                if _BANNER_PATTERN.search(line):
                    issues.append((f"nb:{cid}", lineno, "banner"))
                if _EMOJI_PATTERN.search(line):
                    issues.append((f"nb:{cid}", lineno, "emoji"))
                if _EM_DASH in line or _EN_DASH in line:
                    issues.append((f"nb:{cid}", lineno, "dash"))

    assert not issues, f"AI slop patterns found: {issues}"


def test_no_silent_skip_or_importorskip_in_tests(project_dir: Path) -> None:
    forbidden_attrs = {"importorskip", "skip", "skipif"}
    offenders = []
    for py_file in sorted((project_dir / "tests").rglob("*.py")):
        rel = py_file.relative_to(project_dir).as_posix()
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                owner = node.value
                is_pytest_skip = (
                    (isinstance(owner, ast.Name) and owner.id == "pytest")
                    or (
                        isinstance(owner, ast.Attribute)
                        and owner.attr == "mark"
                        and isinstance(owner.value, ast.Name)
                        and owner.value.id == "pytest"
                    )
                )
                if is_pytest_skip:
                    offenders.append((rel, node.lineno, node.attr))
    assert not offenders, f"Silent skip directives found in tests: {offenders}"


def test_no_unimported_src_modules_or_unused_locals(project_dir: Path) -> None:
    py_files = sorted(project_dir.rglob("*.py"))
    corpus = {p.relative_to(project_dir).as_posix(): p.read_text(encoding="utf-8") for p in py_files}
    nb_path = project_dir / "01_Qwen3_Evaluation_Runner.ipynb"
    if nb_path.is_file():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        corpus[nb_path.name] = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    orphan_modules = []
    for py_file in sorted((project_dir / "src" / "qwen3_eval").rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        rel = py_file.relative_to(project_dir).as_posix()
        mod_name = rel[len("src/") : -3].replace("/", ".")
        pkg_name, stem = mod_name.rsplit(".", 1)
        from_pat = re.compile(rf"from\s+{re.escape(pkg_name)}\s+import\s+.*\b{re.escape(stem)}\b")
        imported_anywhere = any(
            (mod_name in txt or bool(from_pat.search(txt)))
            for other, txt in corpus.items()
            if other != rel
        )
        if not imported_anywhere:
            orphan_modules.append(rel)

    unused_locals = []
    targets = sorted(list((project_dir / "src").rglob("*.py")) + list((project_dir / "scripts").rglob("*.py")))
    for py_file in targets:
        rel = py_file.relative_to(project_dir).as_posix()
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and node.body[0].value.value is Ellipsis
            ):
                continue
            params = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            if node.args.vararg:
                params.append(node.args.vararg.arg)
            if node.args.kwarg:
                params.append(node.args.kwarg.arg)
            assigned = {
                c.id for c in ast.walk(node) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)
            }
            loaded = {
                c.id for c in ast.walk(node) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load)
            }
            for param in params:
                if param not in {"self", "cls"} and not param.startswith("_") and param not in loaded:
                    unused_locals.append((rel, node.lineno, node.name, f"param:{param}"))
            for var in sorted(assigned):
                if not var.startswith("_") and var not in loaded:
                    unused_locals.append((rel, node.lineno, node.name, f"local:{var}"))

    assert not orphan_modules, f"Unimported src modules found: {orphan_modules}"
    assert not unused_locals, f"Unused parameters or locals found: {unused_locals}"


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
        "import qwen3_eval.utils.cache; import qwen3_eval.data.labels; "
        "import qwen3_eval.metrics.contingency; "
        "heavy = [m for m in ('torch', 'transformers', 'statsmodels') if m in sys.modules]; "
        "assert not heavy, f'Unexpected heavy imports: {heavy}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
