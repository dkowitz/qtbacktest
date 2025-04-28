"""UI widgets related to strategy selection and parameter editing."""

from __future__ import annotations

from typing import Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QLineEdit,
)

from qt_prototype.strategy_loader import STRATEGY_MANAGER


class StrategyDockWidget(QWidget):
    """Content widget placed inside the Strategy dock."""

    def __init__(self, parent=None):
        super().__init__(parent)

        vlayout = QVBoxLayout(self)
        vlayout.setContentsMargins(4, 4, 4, 4)

        # Combobox to pick strategy -------------------------------------
        self._combo = QComboBox()
        self._combo.addItems(STRATEGY_MANAGER.names())
        self._combo.currentTextChanged.connect(self._rebuild_params)
        vlayout.addWidget(self._combo)

        # Form layout for parameter widgets -----------------------------
        self._form = QFormLayout()
        vlayout.addLayout(self._form)

        # Run button (prints params for now) ----------------------------
        self._run_btn = QPushButton("Run (F5)")
        self._run_btn.clicked.connect(self._emit_run)
        vlayout.addWidget(self._run_btn)

        self._rebuild_params(self._combo.currentText())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_params(self, strategy_name: str):
        # Clear old widgets
        while self._form.rowCount():
            self._form.removeRow(0)

        cls = STRATEGY_MANAGER.get(strategy_name)
        schema: Dict[str, Any] = getattr(cls, "param_schema", {})

        self._param_widgets: Dict[str, Any] = {}

        for key, info in schema.items():
            if isinstance(info, tuple):
                dtype, default, min_, max_ = info
            else:
                dtype = info.get("type", int)
                default = info.get("default", 0)
                min_ = info.get("min", 0)
                max_ = info.get("max", 100)

            if dtype is int:
                w = QSpinBox()
                w.setRange(min_, max_)
                w.setValue(default)
            elif dtype is float:
                w = QDoubleSpinBox()
                w.setRange(min_, max_)
                w.setDecimals(4)
                w.setValue(default)
            elif dtype is bool:
                w = QCheckBox()
                w.setChecked(bool(default))
            else:
                # Fallback to string spinbox-less
                from PySide6.QtWidgets import QLineEdit

                w = QLineEdit(str(default))

            self._param_widgets[key] = w
            self._form.addRow(key, w)

    # Signal emitted when user hits Run ---------------------------------

    run_requested = Signal(object, dict)  # (strategy_cls, params)

    # ------------------------------------------------------------------
    def _emit_run(self):
        params = {}
        for key, widget in self._param_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                params[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            else:
                params[key] = widget.text()

        strategy_name = self._combo.currentText()
        cls = STRATEGY_MANAGER.get(strategy_name)
        self.run_requested.emit(cls, params)
