"""Import d'un script de scripts/ comme module (les scripts ne sont pas un paquet)."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from mikata.config import PROJECT_ROOT


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # requis par @dataclass
    spec.loader.exec_module(module)
    return module
