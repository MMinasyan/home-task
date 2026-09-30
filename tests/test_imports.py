"""Import-direction checks for the package sources."""

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
STDLIB = frozenset(sys.stdlib_module_names)


def imported_modules(path):
    """Top-level or fully dotted module names imported anywhere in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = list(path.relative_to(SRC).parent.parts)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                modules.add(node.module or "")
            else:
                base = package[: len(package) - (node.level - 1)]
                tail = node.module.split(".") if node.module else []
                modules.add(".".join(base + tail))
    return modules


def provider_module_allowed(module):
    """Standard library, pydantic, httpx, or this package's model/providers areas."""
    parts = module.split(".")
    if parts[0] in STDLIB or parts[0] in ("pydantic", "httpx"):
        return True
    if parts[0] == "agent_qa":
        return parts[1:2] in (["model"], ["providers"], [])
    return False


def test_package_initializers_exist_and_are_empty():
    for relative in ("agent_qa/__init__.py", "agent_qa/providers/__init__.py"):
        path = SRC / relative
        assert path.is_file(), relative
        assert path.read_text(encoding="utf-8") == "", relative


def test_model_module_imports_standard_library_only():
    modules = imported_modules(SRC / "agent_qa" / "model.py")
    outside = sorted(m for m in modules if m.split(".")[0] not in STDLIB)
    assert outside == [], outside


def test_storage_module_imports_standard_library_only():
    modules = imported_modules(SRC / "agent_qa" / "storage.py")
    outside = sorted(m for m in modules if m.split(".")[0] not in STDLIB)
    assert outside == [], outside


@pytest.mark.parametrize(
    "relative",
    ["agent_qa/providers/_cc.py", "agent_qa/providers/_sse.py"],
)
def test_provider_module_imports_standard_library_and_model_only(relative):
    modules = imported_modules(SRC / relative)
    outside = sorted(m for m in modules if not (m.split(".")[0] in STDLIB or m == "agent_qa.model"))
    assert outside == [], outside


def test_provider_modules_import_only_their_dependencies():
    for path in sorted((SRC / "agent_qa" / "providers").glob("*.py")):
        outside = sorted(m for m in imported_modules(path) if not provider_module_allowed(m))
        assert outside == [], f"{path.name}: {outside}"


def test_client_module_imports_only_the_transport_dependencies():
    modules = imported_modules(SRC / "agent_qa" / "providers" / "chat_completions.py")
    allowed = frozenset(
        {
            "httpx",
            "agent_qa.model",
            "agent_qa.providers.config",
            "agent_qa.providers._cc",
            "agent_qa.providers._sse",
        }
    )
    outside = sorted(m for m in modules if m.split(".")[0] not in STDLIB and m not in allowed)
    assert outside == [], outside


def test_turns_module_imports_only_the_storage_model_and_provider_dependencies():
    modules = imported_modules(SRC / "agent_qa" / "turns.py")
    allowed = frozenset(
        {
            "agent_qa.model",
            "agent_qa.storage",
            "agent_qa.providers.chat_completions",
            "agent_qa.providers.config",
        }
    )
    outside = sorted(m for m in modules if m.split(".")[0] not in STDLIB and m not in allowed)
    assert outside == [], outside
