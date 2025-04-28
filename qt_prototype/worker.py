"""Background thread that executes a Strategy over a DataFrame."""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from PySide6.QtCore import QThread, Signal

from .result_registry import BacktestResult


class BacktestWorker(QThread):
    """Very rudimentary back-test loop – placeholder for future engine."""

    finished = Signal(dict)  # results dict

    def __init__(self, strategy_cls, params: Dict[str, Any], dataset_name: str, df: pd.DataFrame):
        super().__init__()
        self.strategy_cls = strategy_cls
        self.params = params
        self.df = df
        self.dataset_name = dataset_name

    # ------------------------------------------------------------------
    def run(self):  # noqa: D401 – Qt entrypoint
        # Work on a *lower-case column* copy so strategies can rely on
        # canonical names regardless of csv heading capitalisation.
        df = self.df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Basic validation: need at least a 'close' column
        if "close" not in df.columns:
            self.finished.emit({"error": "Dataset lacks 'Close' column"})
            return

        strat = self.strategy_cls(**self.params)
        strat.prepare(df)

        equity = []
        cash = 10000.0
        position = 0  # shares

        for i, row in df.iterrows():
            price = row["close"]
            signal = strat.next(i, row, position)

            if signal == "BUY" and position == 0:
                position = cash / price
                cash = 0
            elif signal == "SELL" and position > 0:
                cash = position * price
                position = 0

            equity.append(cash + position * price)

        bt_res = BacktestResult(
            id=-1,
            strategy_name=self.strategy_cls.__name__,
            dataset_name=self.dataset_name,
            params=self.params,
            equity=pd.Series(equity),
            final_balance=equity[-1] if equity else None,
        )

        self.finished.emit({"result": bt_res})
