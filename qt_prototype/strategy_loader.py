"""Dynamic discovery of strategy plug-ins in the *strategies* package."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from typing import Dict


class StrategyManager:
    """Find subclasses of ``strategies.base.Strategy`` available on disk."""

    def __init__(self):
        from pathlib import Path  # local import to avoid polluting global namespace

        self._strategies: Dict[str, type] = {}
        self._load_strategies()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def names(self):
        return list(self._strategies.keys())

    def get(self, name: str):
        return self._strategies[name]

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _load_strategies(self):
        pkg_name = "strategies"
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or spec.submodule_search_locations is None:
            return

        for _, modname, ispkg in pkgutil.iter_modules(spec.submodule_search_locations):
            if ispkg or modname.startswith("__"):
                continue
            full_name = f"{pkg_name}.{modname}"
            module = importlib.import_module(full_name)

            from strategies.base import Strategy  # type: ignore

            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                    self._strategies[obj.__name__] = obj


STRATEGY_MANAGER = StrategyManager()
