"""Scan gui/*.py for tr("...{...}...") templates missing from _ZH2EN (temp tool)."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "sharp3d" / "gui"


def zh2en_keys() -> set[str]:
    src = (ROOT / "i18n.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        for stmt in ast.walk(node):
            pass
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "_ZH2EN":
            return {k.value for k in node.value.keys}
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "_ZH2EN":
            return {k.value for k in node.value.keys}
    raise RuntimeError("_ZH2EN not found")


def tr_templates() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for py in sorted(ROOT.glob("*.py")):
        if py.name == "i18n.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "tr" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                continue
            s = node.args[0].value
            if "{" in s and "}" in s:
                found.setdefault(s, []).append(py.name)
    return found


def main() -> int:
    keys = zh2en_keys()
    templates = tr_templates()
    missing = {t: files for t, files in templates.items() if t not in keys}
    print(f"templates with placeholders: {len(templates)}, "
          f"registered: {len(templates) - len(missing)}, MISSING: {len(missing)}")
    for t, files in sorted(missing.items()):
        print(f"  [{','.join(files)}] {t!r}")
    # also report registered-but-unused placeholder templates (dead entries)
    dead = [k for k in keys if "{" in k and k not in templates]
    if dead:
        print(f"registered but no call site found: {len(dead)}")
        for t in dead:
            print(f"  {t!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
