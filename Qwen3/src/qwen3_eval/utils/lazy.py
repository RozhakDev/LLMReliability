import importlib
from typing import Any, Mapping


def resolve_lazy_export(
    package_name: str,
    attr_name: str,
    export_map: Mapping[str, str],
) -> Any:
    if attr_name in export_map:
        module = importlib.import_module(export_map[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {package_name!r} has no attribute {attr_name!r}")
