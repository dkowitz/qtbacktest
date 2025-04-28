"""Minimal Qt prototype for the trading/back-testing workbench.

Currently implemented features
--------------------------------
1. *Dataset Manager* – keeps all `pandas.DataFrame`s loaded from CSV in
   memory and exposes simple helper methods.
2. *Dataset Navigator* (dock, left) – shows the list of datasets; double
   click to open in a workspace.
3. *Workspace Tabs* (center) – each tab hosts a chart + table view of a
   dataset using a QSplitter so the user can resize the two panes.
4. *Python Console* (dock, bottom) – an embedded IPython console with a
   predefined ``ds`` dict mapping dataset names → DataFrames.

The goal is to provide a concrete, runnable starting point that we can
iterate on – NOT a production-grade application yet.  The code therefore
prefers clarity over edge-case handling or performance optimisation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional

import pandas as pd

# ---- Qt bindings ---------------------------------------------------------
# We prefer PySide6 because it ships the LGPL Qt6 binaries out of the box.
# If you want to use PyQt6 instead, everything is compatible – simply
# change the import below.

from PySide6.QtCore import Qt, QModelIndex  # type: ignore
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QDockWidget,
    QListWidget,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QCheckBox,
    QHBoxLayout,
    QSplitter,
    QTableView,
    QLabel,
    QListWidgetItem,
    QComboBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QSizePolicy,
)

# Local import for editor dock
# Import will be handled in the main function after QApplication is created
HAS_EDITOR = False

# matplotlib toolbar for pan/zoom ---------------------------------------------------

try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:  # pragma: no cover – matplotlib < 3.6 fallback
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar  # type: ignore

# matplotlib embedding -----------------------------------------------------

# Matplotlib < 3.6.0 contains an incompatibility with recent PySide6
# (``KeyboardModifier`` enum no longer converts cleanly to ``int``).  Provide a
# helpful message instead of crashing with a cryptic ``TypeError`` if the user
# runs an old distro-packaged Matplotlib.

import matplotlib as _mpl


def _require_modern_mpl() -> None:  # pragma: no cover – runtime guard only
    major, minor, *_ = map(int, _mpl.__version__.split(".")[:2])
    if (major, minor) < (3, 6):
        raise RuntimeError(
            "Matplotlib >= 3.6 is required for Qt6/PySide6 but version "
            f"{_mpl.__version__} was detected.  Please run\n\n"
            "    pip install --upgrade matplotlib\n\n"
            "or use a virtual environment that contains a newer release."
        )


_require_modern_mpl()

# Matplotlib -----------------------------------------------------------------------
# Import early so we can eventually tweak the global style once we know whether a
# dark theme is active.

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Local import for StrategyDock
from qt_prototype.strategy_ui import StrategyDockWidget

# IPython console embedding -------------------------------------------------

# ---------------------------------------------------------------------
# qtconsole embedding – backwards-compatible import handling
# ---------------------------------------------------------------------

try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget  # type: ignore

    try:
        # qtconsole ≤ 5.5
        from qtconsole.manager import start_new_kernel  # type: ignore
    except ImportError:  # pragma: no cover – new qtconsole ≥ 5.6
        from jupyter_client.manager import start_new_kernel  # type: ignore

    _HAS_QTCONSOLE = True
except ImportError:  # pragma: no cover – qtconsole not present
    _HAS_QTCONSOLE = False


# -------------------------------------------------------------------------
# Backend helpers
# -------------------------------------------------------------------------


class DatasetManager:
    """Singleton-style helper that owns every loaded DataFrame.

    The class also implements a very small *observer* mechanism so that
    interested UI widgets (e.g. the dataset navigator) are notified when a
    new DataFrame is registered.
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._overlays: Dict[str, List["Overlay"]] = {}

        # callbacks taking the newly added dataset *name* as their only arg
        self._listeners: List[Callable[[str], None]] = []
        self._overlay_listeners: List[Callable[[str, "Overlay"], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Registration helpers – notify listeners after success
    # ------------------------------------------------------------------

    def add_csv(self, path: str | Path) -> str:
        """Load *path* into memory and return the assigned dataset name.

        The dataset name is the file stem (without extension).  If a name
        already exists, we append ``_1``, ``_2``… similar to how VS Code
        handles duplicate filenames.
        """

        path = Path(path)
        df = pd.read_csv(path)

        # Derive unique dataset name
        base = path.stem
        name = base
        suffix = 1
        while name in self._datasets:
            name = f"{base}_{suffix}"
            suffix += 1

        self._datasets[name] = df
        self._notify_listeners(name)
        return name

    def add_dataframe(self, df: pd.DataFrame, name: str | None = None) -> str:
        """Register an **already existing** DataFrame in memory.

        Parameters
        ----------
        df
            The DataFrame to register.
        name
            Optional explicit dataset name; if *None* we fall back to the
            generic ``df_X`` counter.
        """

        if name is None or name == "":
            base = "df"
            suffix = 1
            candidate = f"{base}_{suffix}"
            while candidate in self._datasets:
                suffix += 1
                candidate = f"{base}_{suffix}"
            name = candidate

        if name in self._datasets:
            raise ValueError(f"Dataset name '{name}' already exists")

        self._datasets[name] = df
        self._notify_listeners(name)
        return name

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def add_overlay(self, dataset_name: str, overlay: "Overlay") -> None:
        self._overlays.setdefault(dataset_name, []).append(overlay)
        self._notify_overlay_listeners(dataset_name, overlay)

    def overlays(self, dataset_name: str) -> List["Overlay"]:
        return self._overlays.get(dataset_name, [])

    def get(self, name: str) -> pd.DataFrame:
        return self._datasets[name]

    def names(self):  # noqa: D401 – simple iterator helper
        """Yield dataset names (order corresponds to insertion order)."""
        return self._datasets.keys()

    # Allow dictionary-like access --------------------------------------------------

    def __getitem__(self, key: str) -> pd.DataFrame:  # noqa: DunderMethods
        return self._datasets[key]

    def __iter__(self):  # noqa: DunderMethods
        return iter(self._datasets)

    # ------------------------------------------------------------------
    # Observer registration
    # ------------------------------------------------------------------

    def add_listener(self, fn: Callable[[str], None]):
        """Register *fn* to be called whenever a new dataset is added."""

        if fn not in self._listeners:
            self._listeners.append(fn)

    def add_overlay_listener(self, fn: Callable[[str, "Overlay"], None]):
        if fn not in self._overlay_listeners:
            self._overlay_listeners.append(fn)

    # Internal -----------------------------------------------------------

    def _notify_listeners(self, name: str):
        for fn in self._listeners:
            try:
                fn(name)
            except Exception:  # pragma: no cover – listeners should not crash app
                import traceback

                traceback.print_exc()

    def _notify_overlay_listeners(self, dataset_name: str, overlay: "Overlay"):
        for fn in self._overlay_listeners:
            try:
                fn(dataset_name, overlay)
            except Exception:
                import traceback

                traceback.print_exc()


DS_MANAGER = DatasetManager()  # global instance for now


# -------------------------------------------------------------------------
# Overlays
# -------------------------------------------------------------------------


from dataclasses import dataclass, field


@dataclass
class Overlay:
    name: str
    df: pd.DataFrame  # expects columns: start_idx, end_idx, low_val, high_val, dir
    style: dict = field(default_factory=dict)
    visible: bool = True


# -------------------------------------------------------------------------
# Strategy discovery helper
# -------------------------------------------------------------------------


# Strategy loader ---------------------------------------------------------
# The absolute import works when the application is launched with
# ``python -m qt_prototype``.  When users run ``python qt_prototype/main.py``
# directly, the package name is not on `sys.path`, so we fall back to a
# relative import in that case.

try:
    from qt_prototype.strategy_loader import STRATEGY_MANAGER  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – fallback for direct script run
    from strategy_loader import STRATEGY_MANAGER  # type: ignore


# -------------------------------------------------------------------------
# Public helper so that console users can register new DataFrames easily
# -------------------------------------------------------------------------


def register(df: pd.DataFrame, name: str | None = None) -> str:  # noqa: N802 – keep snake_case for Python users
    """Add *df* to the global :data:`DS_MANAGER` and return the dataset name.

    This is the one-liner that users will call in the embedded IPython
    console:

    >>> new = generate_my_indicator(ds["eurusd_5m"])
    >>> register(new, "eurusd_5m_indic")
    'eurusd_5m_indic'
    """

    return DS_MANAGER.add_dataframe(df, name)


def register_overlay(dataset_name: str, name: str, df: pd.DataFrame, style: Optional[dict] = None):
    """Convenience wrapper exposed to the IPython console."""

    DS_MANAGER.add_overlay(dataset_name, Overlay(name, df, style or {}))


# -------------------------------------------------------------------------
# Model for QTableView – minimal Pandas → Qt adapter
# -------------------------------------------------------------------------

# NOTE: Older distributions may ship an ancient pandas version that lacks
# the ``Styler`` class under ``pandas.io.formats.style``.  We do **not** rely
# on Styler; instead we create a small QAbstractTableModel on the fly in
# ``_create_pandas_model``.  The placeholder class below exists merely to
# reserve a spot should we decide to add richer styling later.  For now it
# is an empty stub so that import errors do not propagate.


class PandasModel:  # pragma: no cover – reserved for future use
    pass

# -------------------------------------------------------------------------
# Widgets
# -------------------------------------------------------------------------


class ChartWidget(FigureCanvas):
    """Simple matplotlib chart showing the *Close* column as a line plot."""

    def __init__(self, df: pd.DataFrame, dataset_name: str | None = None, parent: QWidget | None = None):
        self._df = df
        self.dataset_name = dataset_name

        # Matplotlib figure -----------------------------------------------------------------
        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)

        self._ax = self._fig.add_subplot(111)

        # State
        self._chart_mode: str = "Line"  # or "Candlestick"

        # Default columns: Close if present else first numeric
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if "Close" in df.columns:
            self._columns = ["Close"]
        elif numeric_cols:
            self._columns = [numeric_cols[0]]
        else:
            self._columns = []

        self._draw()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def set_chart_mode(self, mode: str):
        if mode != self._chart_mode:
            self._chart_mode = mode
            self._draw()

    def set_columns(self, cols: List[str]):
        self._columns = cols or [self._columns[0]]
        self._draw()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw(self):
        self._ax.clear()

        if self._chart_mode == "Candlestick" and self._has_ohlc():
            self._plot_candles()
        else:
            for col in self._columns:
                if col in self._df.columns:
                    self._ax.plot(self._df[col].values, label=col)
            if len(self._columns) > 1:
                self._ax.legend(loc="upper left", fontsize="small")

        self._ax.set_title(
            ", ".join(self._columns)
            if self._chart_mode == "Line"
            else "OHLC candlestick"
        )
        self._ax.grid(True, linestyle=":", alpha=0.4)

        # Overlays -------------------------------------------------------
        if self.dataset_name is not None:
            for ov in DS_MANAGER.overlays(self.dataset_name):
                if not ov.visible:
                    continue
                for _, seg in ov.df.iterrows():
                    start = seg["start_idx"]
                    end = seg["end_idx"]
                    low = seg["low_val"]
                    high = seg["high_val"]
                    color = "green" if seg.get("dir", "bull") == "bull" else "red"
                    self._ax.axhspan(
                        low,
                        high,
                        xmin=start / len(self._df),
                        xmax=end / len(self._df),
                        facecolor=color,
                        alpha=ov.style.get("alpha", 0.2),
                        linewidth=0,
                    )

        self.draw_idle()

    # ------------------------------------------------------------------
    # Candlestick helpers
    # ------------------------------------------------------------------

    def _find_ohlc_cols(self):
        cols_lower = {c.lower(): c for c in self._df.columns}
        required = ["open", "high", "low", "close"]
        if all(r in cols_lower for r in required):
            return [cols_lower[r] for r in required]
        return None

    def _has_ohlc(self) -> bool:
        return self._find_ohlc_cols() is not None

    def _plot_candles(self):
        try:
            from mplfinance.original_flavor import candlestick_ohlc  # type: ignore
            import matplotlib.dates as mdates

            ohlc_cols = self._find_ohlc_cols()
            if ohlc_cols is None:
                raise RuntimeError("No OHLC columns found")

            ohlc = self._df.loc[:, ohlc_cols].copy()

            if isinstance(ohlc.index, pd.DatetimeIndex):
                ohlc.index = mdates.date2num(ohlc.index.to_pydatetime())
            else:
                ohlc.index = ohlc.index.astype(float)

            oc, hc, lc, cc = ohlc_cols  # original-case names
            data = [
                (idx, row[oc], row[hc], row[lc], row[cc])
                for idx, row in ohlc.iterrows()
            ]

            candlestick_ohlc(self._ax, data, width=0.6, colorup="g", colordown="r", alpha=0.8)
            self._ax.xaxis.set_visible(False)
        except ImportError:
            # Fall back to Close line if mplfinance is not available
            self._ax.plot(self._df["Close"].values, label="Close")


class Workspace(QWidget):
    """A workspace shows one dataset in both chart & table form."""

    def __init__(self, dataset_name: str, parent: QWidget | None = None):
        super().__init__(parent)

        self.dataset_name = dataset_name

        df = DS_MANAGER[dataset_name]

        # ------------------------------------------------------------------
        # Controls row ------------------------------------------------------
        # ------------------------------------------------------------------

        controls = QWidget(self)
        hlayout = QHBoxLayout(controls)
        hlayout.setContentsMargins(2, 2, 2, 2)
        from PySide6.QtWidgets import QSizePolicy
        controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Chart mode selector ----------------------------------------------
        hlayout.addWidget(QLabel("Chart:"))
        mode_combo = QComboBox()
        mode_combo.addItems(["Line", "Candlestick"])
        hlayout.addWidget(mode_combo)

        # Show/Hide table checkbox ---------------------------------------
        table_chk = QCheckBox("Table")
        table_chk.setChecked(True)
        hlayout.addWidget(table_chk)

        # Columns checklist ------------------------------------------------
        col_list = QListWidget()
        col_list.setMaximumHeight(60)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                item = QListWidgetItem(col)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                # Pre-check Close or first numeric column
                if col in ("Close",) or col == df.columns[0]:
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
                col_list.addItem(item)

        hlayout.addWidget(QLabel("Columns:"))
        hlayout.addWidget(col_list, 1)


        # ------------------------------------------------------------------
        # Main splitter  (chart + table)------------------------------------
        # ------------------------------------------------------------------

        splitter = QSplitter(Qt.Vertical)

        # Top – chart -------------------------------------------------------
        chart = ChartWidget(df, dataset_name, splitter)
        splitter.addWidget(chart)

        # Navigation toolbar (pan/zoom) -----------------------------------
        toolbar = NavigationToolbar(chart, controls)
        hlayout.addWidget(toolbar)

        # Overlays checklist ---------------------------------------------
        overlay_list = QListWidget()
        overlay_list.setMaximumHeight(60)

        def _populate_overlays(dataset_name=dataset_name):
            overlay_list.clear()
            for ov in DS_MANAGER.overlays(dataset_name):
                item = QListWidgetItem(ov.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if ov.visible else Qt.Unchecked)
                overlay_list.addItem(item)

        _populate_overlays()

        def _overlay_item_changed(item):
            name = item.text()
            for ov in DS_MANAGER.overlays(dataset_name):
                if ov.name == name:
                    ov.visible = item.checkState() == Qt.Checked
                    chart._draw()
                    break

        overlay_list.itemChanged.connect(_overlay_item_changed)

        # Listen for new overlays added later

        def _on_new_overlay(ds_name, overlay):
            if ds_name == dataset_name:
                _populate_overlays()
                chart._draw()

        DS_MANAGER.add_overlay_listener(_on_new_overlay)

        hlayout.addWidget(QLabel("Overlays:"))
        hlayout.addWidget(overlay_list)

        # Bottom – table ----------------------------------------------------
        table = QTableView(splitter)
        model = _create_pandas_model(df)
        table.setModel(model)
        table.setSortingEnabled(True)
        splitter.addWidget(table)

        # ------------------------------------------------------------------
        # Signals                                                            
        # ------------------------------------------------------------------

        def _update_columns():
            cols = [col_list.item(i).text() for i in range(col_list.count()) if col_list.item(i).checkState() == Qt.Checked]
            chart.set_columns(cols)

        col_list.itemChanged.connect(_update_columns)

        def _update_mode(text):
            chart.set_chart_mode(text)

        mode_combo.currentTextChanged.connect(_update_mode)

        # Show/hide table toggle ------------------------------------------
        def _toggle_table(state):
            table.setVisible(state == Qt.Checked)

        table_chk.stateChanged.connect(_toggle_table)

        # ------------------------------------------------------------------
        # Final layout ------------------------------------------------------
        # ------------------------------------------------------------------

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(splitter)
        self.setLayout(layout)


def _create_pandas_model(df: pd.DataFrame):
    """Return a minimal QAbstractTableModel wrapping *df*."""

    from PySide6.QtCore import QAbstractTableModel  # local import to avoid polluting header

    class _Model(QAbstractTableModel):
        def rowCount(self, *_):
            return len(df)

        def columnCount(self, *_):
            return len(df.columns)

        def data(self, index: QModelIndex, role=Qt.DisplayRole):
            if not index.isValid() or role != Qt.DisplayRole:
                return None
            return str(df.iat[index.row(), index.column()])

        def headerData(self, section, orientation, role=Qt.DisplayRole):
            if role != Qt.DisplayRole:
                return None
            if orientation == Qt.Horizontal:
                return str(df.columns[section])
            return str(df.index[section])

    return _Model()


# -------------------------------------------------------------------------
# Main window
# -------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Back-test Workbench – Prototype")
        self.resize(1200, 800)

        # Central tab widget ------------------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self._tabs)

        # Dataset navigator -------------------------------------------------------
        self._nav_dock = QDockWidget("Datasets", self)
        self._nav_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._nav_list = QListWidget()
        self._nav_list.itemDoubleClicked.connect(self._open_dataset_in_tab)
        self._nav_dock.setWidget(self._nav_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._nav_dock)

        # Listen to dataset additions so navigator stays in sync
        DS_MANAGER.add_listener(self._on_dataset_added)

        # Python console ----------------------------------------------------------
        if _HAS_QTCONSOLE:
            try:
                self._console_dock = QDockWidget("Python Console", self)
                self._console_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)

                self._console_widget = _create_ipython_console(
                    {
                        'ds': DS_MANAGER,
                        'register': register,
                        'register_overlay': register_overlay,
                        'Overlay': Overlay,
                    }
                )

                self._console_dock.setWidget(self._console_widget)
                self.addDockWidget(Qt.BottomDockWidgetArea, self._console_dock)
            except Exception as e:
                import traceback
                print(f"Failed to initialize IPython console: {e}")
                traceback.print_exc()
                self._console_dock = None
                self._console_widget = None
                
        # Code Editor -------------------------------------------------------------
        # Import editor module here when we know QApplication exists
        try:
            global HAS_EDITOR
            
            try:
                from qt_prototype.fallback_editor import EditorDockWidget  # type: ignore
            except ImportError:  # fallback for direct script run
                try:
                    from .fallback_editor import EditorDockWidget  # type: ignore
                except ImportError:
                    import sys
                    import os
                    # Add current directory to path if running as script
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.append(current_dir)
                    from fallback_editor import EditorDockWidget  # type: ignore
                    
            self._editor_dock = QDockWidget("Strategy Editor", self)
            self._editor_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
            
            self._editor_widget = EditorDockWidget(self)
            self._editor_dock.setWidget(self._editor_widget)
            
            # Position the editor next to the console in the bottom dock area 
            # instead of tabbing with it - this allows seeing both at once
            if _HAS_QTCONSOLE and self._console_dock is not None:
                self.addDockWidget(Qt.BottomDockWidgetArea, self._editor_dock)
                self.splitDockWidget(self._console_dock, self._editor_dock, Qt.Horizontal)
            else:
                self.addDockWidget(Qt.BottomDockWidgetArea, self._editor_dock)
                
            global HAS_EDITOR
            HAS_EDITOR = True
            
        except Exception as e:
            import traceback
            print(f"Failed to initialize editor: {e}")
            traceback.print_exc()
            self._editor_dock = None
            self._editor_widget = None

        # Menu & actions ----------------------------------------------------------
        open_act = QAction("&Load CSV…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._load_csv)

        toggle_console_act = QAction("Toggle &Console", self)
        toggle_console_act.setShortcut("F4")
        toggle_console_act.triggered.connect(self._toggle_console)
        
        toggle_editor_act = QAction("Toggle &Editor", self)
        toggle_editor_act.setShortcut("F3")
        toggle_editor_act.triggered.connect(self._toggle_editor)
        
        new_strategy_act = QAction("&New Strategy", self)
        new_strategy_act.setShortcut("Ctrl+N")
        new_strategy_act.triggered.connect(self._new_strategy)
        
        edit_strategy_act = QAction("&Edit Strategy", self)
        edit_strategy_act.triggered.connect(self._edit_strategy)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        file_menu.addAction(new_strategy_act)
        file_menu.addAction(edit_strategy_act)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(toggle_console_act)
        view_menu.addAction(toggle_editor_act)

        # ------------------------------------------------------
        # Strategy dock
        # ------------------------------------------------------

        try:
            from qt_prototype.strategy_ui import StrategyDockWidget  # type: ignore
        except ModuleNotFoundError:  # running as script
            from strategy_ui import StrategyDockWidget  # type: ignore

        strat_widget = StrategyDockWidget(self)
        strat_widget.run_requested.connect(self._run_strategy)

        self._strategy_dock = QDockWidget("Strategies", self)
        self._strategy_dock.setWidget(strat_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self._strategy_dock)

        # Results dock ---------------------------------------------------
        try:
            from qt_prototype.results_ui import ResultsDockWidget  # type: ignore
        except ModuleNotFoundError:
            from results_ui import ResultsDockWidget  # type: ignore

        self._results_widget = ResultsDockWidget(self)
        self._results_widget.open_requested.connect(self._show_result_tab)
        self._results_dock = QDockWidget("Results", self)
        self._results_dock.setWidget(self._results_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._results_dock)

    # ------------------------------------------------------------------
    # Slots / callbacks
    # ------------------------------------------------------------------

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", str(Path.home()), "CSV files (*.csv)")
        if not path:
            return

        DS_MANAGER.add_csv(path)  # navigator will update via listener

    def _open_dataset_in_tab(self, item):  # QListWidgetItem
        name = item.text()
        workspace = Workspace(name)
        self._tabs.addTab(workspace, name)
        self._tabs.setCurrentWidget(workspace)

    # ------------------------------------------------------------------
    # Strategy execution glue
    # ------------------------------------------------------------------

    def _run_strategy(self, strategy_cls, params):
        current = self._tabs.currentWidget()
        if not isinstance(current, Workspace):
            return

        dataset_name = current.dataset_name
        df = DS_MANAGER[dataset_name]

        from qt_prototype.worker import BacktestWorker

        self._worker = BacktestWorker(strategy_cls, params, dataset_name, df)
        self._worker.finished.connect(self._on_backtest_finished)
        self._worker.start()

    def _on_backtest_finished(self, result):
        from PySide6.QtWidgets import QMessageBox
        from qt_prototype.result_registry import RESULTS

        if "error" in result:
            QMessageBox.warning(self, "Back-test error", result["error"])
            return

        bt_res = result["result"]
        id_ = RESULTS.add(bt_res)
        self._results_widget.add_result(bt_res)

    def _show_result_tab(self, id_: int):
        # Deferred imports to avoid circulars / fallback to script mode
        try:
            from qt_prototype.result_tab import ResultTab  # type: ignore
        except ModuleNotFoundError:
            from result_tab import ResultTab  # type: ignore

        try:
            from qt_prototype.result_registry import RESULTS  # type: ignore
        except ModuleNotFoundError:
            from result_registry import RESULTS  # type: ignore

        res = RESULTS.get(id_)
        tab = ResultTab(res)
        self._tabs.addTab(tab, f"Result {id_}: {res.strategy_name}")
        self._tabs.setCurrentWidget(tab)

    def _on_dataset_added(self, name: str):
        """Callback from :pyclass:`DatasetManager` whenever a new DataFrame is registered."""

        # Avoid duplicates in the list widget
        if not any(self._nav_list.item(i).text() == name for i in range(self._nav_list.count())):
            self._nav_list.addItem(name)

    # ------------------------------------------------------------------
    # Console helpers
    # ------------------------------------------------------------------

    def _toggle_console(self):
        if not _HAS_QTCONSOLE:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Console unavailable", "qtconsole is not installed – run 'pip install qtconsole' and restart.")
            return
            
        if self._console_dock is None:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Console unavailable", "Failed to initialize IPython console. Check console output for details.")
            return

        visible = self._console_dock.isVisible()
        self._console_dock.setVisible(not visible)

    def _toggle_editor(self):
        if self._editor_dock is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Editor unavailable", "Failed to initialize code editor. Check console output for details.")
            return
            
        visible = self._editor_dock.isVisible()
        self._editor_dock.setVisible(not visible)

    # Add a new method to handle tab closing
    def _close_tab(self, index):
        """Close the tab at the given index."""
        self._tabs.removeTab(index)

    # ------------------------------------------------------------------
    # Strategy editor integration
    # ------------------------------------------------------------------
    
    def _new_strategy(self):
        """Create a new strategy file in the editor."""
        
        if self._editor_dock is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Editor unavailable", "Failed to initialize code editor. Check console output for details.")
            return
        
        template = '''from strategies.base import Strategy


class MyNewStrategy(Strategy):
    """
    A simple strategy template.
    
    This is a starting point for creating your own trading strategy.
    """

    # Define parameters that can be adjusted in the UI
    param_schema = {
        'fast': (int, 10, 2, 50),    # (type, default, min, max)
        'slow': (int, 30, 5, 100),
        'threshold': (float, 0.001, 0.0, 0.01),
    }

    def __init__(self, **params):
        super().__init__(**params)
        # Initialize any instance variables here
        
    def prepare(self, df):
        """Pre-compute indicators before the bar-by-bar loop."""
        # Calculate indicators on the DataFrame
        pass
        
    def next(self, i, row, position):
        """Process each bar in the backtest.
        
        Parameters:
        -----------
        i : int
            Current bar index
        row : pandas.Series
            Current data row
        position : float
            Current position (negative for short, positive for long, 0 for flat)
            
        Returns:
        --------
        float
            The new desired position
        """
        # Your trading logic goes here
        return 0.0  # No position by default
'''
        self._editor_widget.editor.set_text(template)
        self._editor_dock.show()
        self._editor_dock.raise_()
        
    def _edit_strategy(self):
        """Open an existing strategy file for editing."""
        if self._editor_dock is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Editor unavailable", "Failed to initialize code editor. Check console output for details.")
            return
            
        # Look for strategies directory relative to the current file
        import os
        from pathlib import Path
        
        # Try to find the strategies directory
        possible_paths = [
            Path("strategies"),
            Path("btest/strategies"),
            Path(__file__).parent.parent / "strategies",
        ]
        
        strategy_dir = None
        for path in possible_paths:
            if path.exists() and path.is_dir():
                strategy_dir = path
                break
                
        if strategy_dir is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, 
                "Strategy Directory Not Found", 
                "Could not locate the 'strategies' directory. Please manually navigate to your strategy file."
            )
            initial_dir = str(Path.home())
        else:
            initial_dir = str(strategy_dir)
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Strategy File", initial_dir,
            "Python Files (*.py);;All Files (*.*)"
        )
        
        if file_path:
            self._editor_widget.editor.open_file(file_path)
            self._editor_dock.show()
            self._editor_dock.raise_()


# -------------------------------------------------------------------------
# Helper – IPython console
# -------------------------------------------------------------------------


def _create_ipython_console(namespace: dict):
    """Return a RichJupyterWidget with *namespace* injected into user_ns."""

    console = RichJupyterWidget()
    kernel_manager = None
    kernel_client = None

    # Dark console style if the application palette is dark --------------
    try:
        from qtconsole import styles as _qstyles  # noqa: WPS433
        console.set_default_style("linux")  # dark background scheme
    except Exception:
        pass

    # ------------------------------------------------------------------
    # First choice: in-process kernel (shares memory, easiest to push vars)
    # ------------------------------------------------------------------

    try:
        from qtconsole.inprocess import QtInProcessKernelManager  # type: ignore

        km = QtInProcessKernelManager()
        km.start_kernel(show_banner=False)
        kernel = km.kernel  # access to push user namespace
        kernel.gui = "qt"
        kernel.shell.push(namespace)

        kc = km.client()
        kc.start_channels()

        console.kernel_manager = km
        console.kernel_client = kc
        
        console.banner = (
            "Back-test Workbench (IPython)\n\n"
            "Available objects:\n"
            "  ds                 → DatasetManager\n"
            "  register(df, name) → add DataFrame\n"
            "  register_overlay(dataset, name, df) → add overlay layer\n"
        )
        
        return console

    except Exception:
        # Fallback: external kernel via jupyter_client – variables will be
        # *copied* via pickle, which is slower but still functional.
        try:
            kernel_manager, kernel_client = start_new_kernel()

            # jupyter_client ≥ 8 starts channels automatically; earlier versions
            # require an explicit call.
            try:
                if not kernel_client.shell_channel.is_alive():  # type: ignore[attr-defined]
                    kernel_client.start_channels()
            except AttributeError:
                # Older jupyter_client w/o ``is_alive``
                kernel_client.start_channels()

            # Push variables by executing pickle round-trip inside the separate
            # kernel – this avoids relying on private attributes.
            import base64, cloudpickle  # type: ignore

            payload = base64.b64encode(cloudpickle.dumps(namespace)).decode()
            code = (
                "import base64, cloudpickle, IPython\n"
                f"ns = cloudpickle.loads(base64.b64decode('{payload}'))\n"
                "IPython.get_ipython().user_ns.update(ns)"
            )
            kernel_client.execute(code)
            
            console.kernel_client = kernel_client
            console.kernel_manager = kernel_manager
            
            console.banner = (
                "Back-test Workbench (IPython)\n\n"
                "Available objects:\n"
                "  ds                 → DatasetManager\n"
                "  register(df, name) → add DataFrame\n"
                "  register_overlay(dataset, name, df) → add overlay layer\n"
            )
            
            return console
            
        except Exception:
            # If all fails, return a console with a warning message
            console.banner = "ERROR: Failed to start IPython kernel.\nPlease check your installation of jupyter_client and qtconsole."
            return console


# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------


def main() -> None:  # noqa: D401 – standard name
    app = QApplication(sys.argv)

    # --------------------------------------------------------------
    # Try to enable a dark theme using whichever package is present
    # --------------------------------------------------------------

    import matplotlib as _mpl

    dark_enabled = False

    def _apply_basic_fusion_dark(qapp):
        from PySide6.QtGui import QPalette, QColor

        qapp.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        qapp.setPalette(palette)

    # 1. qdarktheme (>=1.2)
    try:
        import qdarktheme as _qdark  # type: ignore

        if hasattr(_qdark, "setup_theme"):
            _qdark.setup_theme("dark")
            dark_enabled = True
        elif hasattr(_qdark, "load_stylesheet"):
            app.setStyleSheet(_qdark.load_stylesheet())
            app.setPalette(_qdark.load_palette("dark"))
            dark_enabled = True
    except ImportError:
        pass

    # 2. PyQtDarkTheme (older name)
    if not dark_enabled:
        try:
            import PyQtDarkTheme as _pqt  # type: ignore

            if hasattr(_pqt, "setup_theme"):
                _pqt.setup_theme("dark")
            else:
                app.setStyleSheet(_pqt.load_stylesheet())
                app.setPalette(_pqt.load_palette("dark"))
            dark_enabled = True
        except ImportError:
            pass

    # 3. Built-in Fusion fallback
    if not dark_enabled:
        _apply_basic_fusion_dark(app)

    # Sync matplotlib so axes match UI theme
    if dark_enabled:
        try:
            import matplotlib.style  # noqa: WPS433 – runtime import

            _mpl.style.use("dark_background")
        except Exception:
            pass  # Matplotlib too old – skip styling

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
