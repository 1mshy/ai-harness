"""Scenario auto-discovery.

Every ``*.py`` module in this package is imported on first access, which runs
its ``register(Scenario(...))`` side effect. Scenario authors therefore only
ever touch their own file — no shared registry to edit (and no concurrent-write
conflicts when several are built in parallel).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


def load_all() -> None:
    """Import every scenario module so it self-registers.

    Resilient by design: a broken (or, during parallel authoring, half-written)
    scenario module is skipped with a warning instead of taking down the whole
    suite — one bad grader shouldn't hide the others.
    """
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[sales_eval] skipped scenario module {mod.name!r}: "
                             f"{type(e).__name__}: {e}\n")


load_all()
