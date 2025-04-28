"""Central store for completed back-tests during a GUI session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class BacktestResult:
    """Container for everything produced by a single back-test run."""

    id: int
    strategy_name: str
    dataset_name: str
    params: dict

    equity: pd.Series  # indexed like the input DataFrame
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    final_balance: float | None = None


class ResultRegistry:
    """Singleton to keep all results in memory so UI widgets can access them."""

    def __init__(self):
        self._counter: int = 1
        self._results: Dict[int, BacktestResult] = {}

    # ------------------------------------------------------------------
    def add(self, result: BacktestResult) -> int:
        result.id = self._counter
        self._results[self._counter] = result
        self._counter += 1
        return result.id

    def get(self, id_: int) -> BacktestResult:
        return self._results[id_]

    def all(self):
        return list(self._results.values())


RESULTS = ResultRegistry()
