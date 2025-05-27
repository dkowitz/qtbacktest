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
# Removed sys.path printing for cleaner output
# print("--- sys.path at startup ---")
# for p in sys.path:
# print(p)
# print("---------------------------")

from pathlib import Path
from typing import Callable, Dict, List, Any, Optional

import pandas as pd

# ---- Qt bindings ---------------------------------------------------------
# We prefer PySide6 because it ships the LGPL Qt6 binaries out of the box.
# If you want to use PyQt6 instead, everything is compatible – simply
# change the import below.

from PySide6.QtCore import Qt, QModelIndex, QSettings, QSize, QEvent, QObject  # type: ignore
from PySide6.QtGui import QAction, QColor, QBrush, QPalette, QPixmap, QPainter
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
    QMenu,
    QScrollArea,
    QToolButton,
    QFrame,
    QColorDialog,
    QWidgetAction,
    QMessageBox,
    QInputDialog
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

# Local import for live chart
print("Attempting to import EnhancedLiveChartWidget from live_chart_streaming...")
from qt_prototype.live_chart_streaming import EnhancedLiveChartWidget as LiveChartWidget
print("Successfully imported EnhancedLiveChartWidget as LiveChartWidget.")
HAS_LIVE_CHART = True

# Local import for drawing tools
# HAS_DRAWING_TOOLS is now defined in drawing_tools.py
DrawingManager = None         # Define fallbacks in case of import error
CoordinateTransformer = None
DrawingRenderer = None
DrawingToolType = None 
DrawingToolbar = None 
TextInputDialog = None

try:
    from qt_prototype.drawing_tools import (
        HAS_DRAWING_TOOLS,
        DrawingManager as ImportedDrawingManager,
        CoordinateTransformer as ImportedCoordinateTransformer,
        DrawingToolType as ImportedDrawingToolType
    )
    if HAS_DRAWING_TOOLS:
        DrawingManager = ImportedDrawingManager
        CoordinateTransformer = ImportedCoordinateTransformer
        DrawingToolType = ImportedDrawingToolType
        from qt_prototype.drawing_renderer import DrawingRenderer as ImportedDrawingRenderer
        DrawingRenderer = ImportedDrawingRenderer
        # DrawingToolbar and TextInputDialog are for UI elements, import them separately if needed by MainWindow
        from qt_prototype.drawing_toolbar import DrawingToolbar as ImportedDrawingToolbar, TextInputDialog as ImportedTextInputDialog
        DrawingToolbar = ImportedDrawingToolbar
        TextInputDialog = ImportedTextInputDialog
    else:
        # This else might be redundant if HAS_DRAWING_TOOLS correctly reflects import success
        HAS_DRAWING_TOOLS = False # Ensure it's false if inner imports fail somehow
except ImportError:
    HAS_DRAWING_TOOLS = False
    # Fallbacks are already defined above

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
        self._remove_listeners: List[Callable[[str], None]] = []

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
        
    def remove_dataset(self, name: str) -> bool:
        """Remove a dataset from the manager.
        
        Parameters
        ----------
        name
            The name of the dataset to remove.
            
        Returns
        -------
        bool
            True if the dataset was removed, False if it wasn't found.
        """
        if name in self._datasets:
            # Remove the dataset
            del self._datasets[name]
            
            # Remove any associated overlays
            if name in self._overlays:
                del self._overlays[name]
                
            # Notify listeners
            self._notify_remove_listeners(name)
            return True
        return False
        
    def update_dataset(self, name: str, df: pd.DataFrame) -> bool:
        """Update an existing dataset with new data.
        
        Parameters
        ----------
        name
            The name of the dataset to update.
        df
            The new DataFrame to replace the existing one.
            
        Returns
        -------
        bool
            True if the dataset was updated, False if it wasn't found.
        """
        if name in self._datasets:
            self._datasets[name] = df
            self._notify_listeners(name)  # Notify as if it was newly added
            return True
        return False

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
            
    def add_remove_listener(self, fn: Callable[[str], None]):
        """Register *fn* to be called whenever a dataset is removed."""
        if fn not in self._remove_listeners:
            self._remove_listeners.append(fn)

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
                
    def _notify_remove_listeners(self, name: str):
        for fn in self._remove_listeners:
            try:
                fn(name)
            except Exception:
                import traceback
                traceback.print_exc()


DS_MANAGER = DatasetManager()  # global instance for now

# -------------------------------------------------------------------------
# Enhanced IPython Console Helpers
# -------------------------------------------------------------------------

class DatasetAccessor:
    """Accessor class for easier dataset management in the IPython console.
    
    This provides several ways to access datasets:
    - By index: ds[0], ds[1], etc.
    - By alias: ds.a, ds.b, etc. (first 26 datasets mapped to letters)
    - By name: ds["dataset_name"]
    
    It also provides utility methods for dataset operations.
    """
    
    def __init__(self, dataset_manager: DatasetManager):
        self._ds_manager = dataset_manager
        
    def __getitem__(self, key):
        """Get dataset by index, name, or slice."""
        if isinstance(key, int):
            # Access by index
            names = list(self._ds_manager.names())
            if 0 <= key < len(names):
                return self._ds_manager[names[key]]
            else:
                raise IndexError(f"Dataset index {key} out of range (0-{len(names)-1})")
        elif isinstance(key, str):
            # Access by name
            return self._ds_manager[key]
        elif isinstance(key, slice):
            # Return multiple datasets as a list
            names = list(self._ds_manager.names())
            indices = range(*key.indices(len(names)))
            return [self._ds_manager[names[i]] for i in indices]
        else:
            raise KeyError(f"Invalid key type: {type(key)}")
    
    def __getattr__(self, name):
        """Allow attribute access for single-letter aliases (a-z)."""
        if len(name) == 1 and 'a' <= name <= 'z':
            # Map letters to dataset indices
            index = ord(name) - ord('a')
            names = list(self._ds_manager.names())
            if 0 <= index < len(names):
                return self._ds_manager[names[index]]
            else:
                raise AttributeError(f"No dataset at index {index} for alias '{name}'")
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __dir__(self):
        """Make letter aliases available in tab completion."""
        names = list(self._ds_manager.names())
        # Add single-letter aliases for the first 26 datasets
        aliases = [chr(ord('a') + i) for i in range(min(26, len(names)))]
        return super().__dir__() + aliases
    
    def __iter__(self):
        """Iterate over all datasets."""
        return iter(self._ds_manager)
        
    def __len__(self):
        """Return the number of datasets."""
        return len(list(self._ds_manager.names()))
    
    def list(self):
        """List all datasets with their indices and aliases."""
        names = list(self._ds_manager.names())
        result = []
        
        for i, name in enumerate(names):
            alias = chr(ord('a') + i) if i < 26 else '-'
            shape = self._ds_manager[name].shape
            result.append((i, alias, name, shape))
            
        # Create a formatted table
        print("Index | Alias | Name | Shape")
        print("------|-------|------|------")
        for i, alias, name, shape in result:
            print(f"{i:5d} | {alias:5s} | {name:20s} | {shape}")
        
    def names(self):
        """Return a list of all dataset names."""
        return list(self._ds_manager.names())
    
    def get_name(self, index):
        """Get the name of a dataset by index."""
        names = list(self._ds_manager.names())
        if 0 <= index < len(names):
            return names[index]
        else:
            raise IndexError(f"Dataset index {index} out of range (0-{len(names)-1})")
    
    def get_index(self, name):
        """Get the index of a dataset by name."""
        names = list(self._ds_manager.names())
        try:
            return names.index(name)
        except ValueError:
            raise ValueError(f"Dataset '{name}' not found")


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


def update_dataset(target_name: str, source_df: pd.DataFrame) -> bool:
    """Update an existing dataset with new data.
    
    Parameters
    ----------
    target_name : str
        Name of the dataset to update
    source_df : DataFrame
        DataFrame with the new data
        
    Returns
    -------
    bool
        True if the update was successful, False if the target dataset wasn't found
    
    Examples
    --------
    >>> df = ds[0].copy()  # Make a copy of the first dataset
    >>> df['new_column'] = calculations(df)  # Make some changes
    >>> update_dataset('original_name', df)  # Update the original dataset
    True
    """
    return DS_MANAGER.update_dataset(target_name, source_df)


def copy_dataset(source_name: str, new_name: str = None) -> str:
    """Create a copy of an existing dataset.
    
    Parameters
    ----------
    source_name : str
        Name of the dataset to copy
    new_name : str, optional
        Name for the new dataset. If None, a name will be generated
        
    Returns
    -------
    str
        The name of the new dataset
        
    Examples
    --------
    >>> copy_dataset('eurusd_5m', 'eurusd_5m_copy')
    'eurusd_5m_copy'
    >>> copy_dataset('eurusd_5m')  # Auto-generated name
    'eurusd_5m_1'
    """
    try:
        df = DS_MANAGER[source_name].copy()
        if new_name is None:
            # Generate a name based on the source name
            base = f"{source_name}_copy"
            suffix = ""
            candidate = base
            i = 1
            while candidate in DS_MANAGER.names():
                candidate = f"{base}_{i}"
                i += 1
            new_name = candidate
        return register(df, new_name)
    except KeyError:
        raise ValueError(f"Dataset '{source_name}' not found")


def describe_dataset(name_or_index):
    """Print a summary of a dataset.
    
    Parameters
    ----------
    name_or_index : str or int
        Name or index of the dataset
        
    Examples
    --------
    >>> describe_dataset('eurusd_5m')
    # ... prints dataset information ...
    >>> describe_dataset(0)  # First dataset
    # ... prints dataset information ...
    """
    # Get the dataset
    if isinstance(name_or_index, int):
        names = list(DS_MANAGER.names())
        if 0 <= name_or_index < len(names):
            name = names[name_or_index]
            df = DS_MANAGER[name]
        else:
            raise IndexError(f"Dataset index {name_or_index} out of range (0-{len(names)-1})")
    else:
        name = name_or_index
        df = DS_MANAGER[name]
    
    # Print information
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape}")
    print("\nColumn info:")
    
    # Get column information
    for col in df.columns:
        dtype = df[col].dtype
        non_nulls = df[col].count()
        nulls = df[col].isnull().sum()
        unique_vals = df[col].nunique()
        
        print(f"  {col}: {dtype} ({non_nulls} non-null, {nulls} null, {unique_vals} unique values)")
    
    # Print basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print("\nNumeric column statistics:")
        print(df[numeric_cols].describe().to_string())
    
    # Print a few rows as a sample
    print("\nSample (first 5 rows):")
    print(df.head().to_string())


def compare_datasets(name1_or_index1, name2_or_index2):
    """Compare two datasets and display differences.
    
    Parameters
    ----------
    name1_or_index1 : str or int
        Name or index of the first dataset
    name2_or_index2 : str or int
        Name or index of the second dataset
        
    Examples
    --------
    >>> compare_datasets('eurusd_5m', 'eurusd_5m_modified')
    # ... prints comparison information ...
    >>> compare_datasets(0, 1)  # Compare first and second datasets
    # ... prints comparison information ...
    """
    # Get the first dataset
    if isinstance(name1_or_index1, int):
        names = list(DS_MANAGER.names())
        if 0 <= name1_or_index1 < len(names):
            name1 = names[name1_or_index1]
            df1 = DS_MANAGER[name1]
        else:
            raise IndexError(f"Dataset index {name1_or_index1} out of range (0-{len(names)-1})")
    else:
        name1 = name1_or_index1
        df1 = DS_MANAGER[name1]
    
    # Get the second dataset
    if isinstance(name2_or_index2, int):
        names = list(DS_MANAGER.names())
        if 0 <= name2_or_index2 < len(names):
            name2 = names[name2_or_index2]
            df2 = DS_MANAGER[name2]
        else:
            raise IndexError(f"Dataset index {name2_or_index2} out of range (0-{len(names)-1})")
    else:
        name2 = name2_or_index2
        df2 = DS_MANAGER[name2]
    
    # Print basic comparison
    print(f"Comparing '{name1}' and '{name2}':")
    print(f"  Shape of '{name1}': {df1.shape}")
    print(f"  Shape of '{name2}': {df2.shape}")
    
    # Compare columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"\nColumns in both: {len(common_cols)}")
    print(f"Columns only in '{name1}': {len(only_in_1)} {list(only_in_1) if only_in_1 else ''}")
    print(f"Columns only in '{name2}': {len(only_in_2)} {list(only_in_2) if only_in_2 else ''}")
    
    # For common columns, check data type differences
    if common_cols:
        print("\nColumn type differences:")
        type_diff = False
        for col in common_cols:
            if df1[col].dtype != df2[col].dtype:
                print(f"  '{col}': {df1[col].dtype} in '{name1}', {df2[col].dtype} in '{name2}'")
                type_diff = True
        if not type_diff:
            print("  (None)")
    
    # For common columns, check value differences if shapes are the same
    if df1.shape[0] == df2.shape[0] and common_cols:
        print("\nValue differences in common columns:")
        value_diff = False
        for col in common_cols:
            if not df1[col].equals(df2[col]):
                # Count differing values
                diff_count = (df1[col] != df2[col]).sum()
                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    # For numeric columns, calculate stats on differences
                    mean_diff = (df1[col] - df2[col]).mean()
                    max_diff = (df1[col] - df2[col]).abs().max()
                    print(f"  '{col}': {diff_count} differences (mean diff: {mean_diff:.6g}, max abs diff: {max_diff:.6g})")
                else:
                    print(f"  '{col}': {diff_count} differences")
                value_diff = True
        if not value_diff:
            print("  (None)")


def apply_to_dataset(name_or_index, func, columns=None, new_columns=None, result_name=None):
    """Apply a function to columns of a dataset and return a new or updated dataset.
    
    Parameters
    ----------
    name_or_index : str or int
        Name or index of the dataset
    func : callable
        Function to apply to each column. Can either:
        - Take a Series and return a Series
        - Take a DataFrame and return a DataFrame
    columns : list or None
        List of columns to apply the function to. If None, applies to all columns.
    new_columns : list or None
        Names for the new columns. If None, overwrites existing columns.
    result_name : str or None
        Name for the result dataset. If None, auto-generates a name.
        
    Returns
    -------
    str
        Name of the resulting dataset
        
    Examples
    --------
    >>> # Apply normalization to several columns
    >>> apply_to_dataset('prices', lambda x: (x - x.min()) / (x.max() - x.min()), 
    ...                 columns=['Open', 'High', 'Low', 'Close'], 
    ...                 new_columns=['Open_norm', 'High_norm', 'Low_norm', 'Close_norm'],
    ...                 result_name='prices_normalized')
    'prices_normalized'
    
    >>> # Apply moving average to a single column
    >>> apply_to_dataset('prices', lambda x: x.rolling(20).mean(), 
    ...                 columns=['Close'], 
    ...                 new_columns=['MA20'])
    'prices_MA20'
    """
    # Get the dataset
    if isinstance(name_or_index, int):
        names = list(DS_MANAGER.names())
        if 0 <= name_or_index < len(names):
            name = names[name_or_index]
            df = DS_MANAGER[name]
        else:
            raise IndexError(f"Dataset index {name_or_index} out of range (0-{len(names)-1})")
    else:
        name = name_or_index
        df = DS_MANAGER[name]
    
    # Make a copy to work with
    result_df = df.copy()
    
    # If columns is None, use all columns
    if columns is None:
        target_columns = df.columns.tolist()
    else:
        target_columns = columns
        # Verify all columns exist
        for col in target_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset '{name}'")
    
    # Apply function to selected columns
    if new_columns is not None:
        # Ensure new_columns has same length as columns
        if len(new_columns) != len(target_columns):
            raise ValueError(f"Length of new_columns ({len(new_columns)}) must match length of columns ({len(target_columns)})")
        
        # Apply function and create new columns
        for old_col, new_col in zip(target_columns, new_columns):
            try:
                # Try applying to a single Series
                result_df[new_col] = func(df[old_col])
            except Exception as e:
                print(f"Warning: Error applying function to column '{old_col}': {e}")
                print("Trying to apply function to entire DataFrame...")
                try:
                    # Try applying to the whole DataFrame (subset to selected columns)
                    result = func(df[target_columns])
                    if isinstance(result, pd.DataFrame):
                        # If function returns a DataFrame, we'll use its columns
                        for i, (_, series) in enumerate(result.items()):
                            if i < len(new_columns):
                                result_df[new_columns[i]] = series
                    break  # Exit the loop, we've handled all columns at once
                except Exception as e2:
                    raise ValueError(f"Failed to apply function: {e2}")
    else:
        # No new column names provided, overwrite existing columns
        try:
            # Try applying to the DataFrame at once
            subset_df = df[target_columns]
            result = func(subset_df)
            if isinstance(result, pd.DataFrame):
                # Function returned a DataFrame, replace columns
                for col in target_columns:
                    if col in result.columns:
                        result_df[col] = result[col]
            else:
                # Otherwise, try column by column
                for col in target_columns:
                    result_df[col] = func(df[col])
        except Exception as e:
            # Try column by column
            for col in target_columns:
                try:
                    result_df[col] = func(df[col])
                except Exception as e2:
                    raise ValueError(f"Failed to apply function to column '{col}': {e2}")
    
    # Generate result name if not provided
    if result_name is None:
        if new_columns is not None:
            # Use the first new column as an indicator of what was computed
            result_name = f"{name}_{new_columns[0]}"
        else:
            result_name = f"{name}_modified"
    
    # Register the result
    return register(result_df, result_name)


def find_datasets(criteria=None, column_pattern=None, name_pattern=None):
    """Find datasets matching specific criteria.
    
    Parameters
    ----------
    criteria : callable or None
        Function that takes a DataFrame and returns True/False
    column_pattern : str or None 
        Regex pattern to match column names
    name_pattern : str or None
        Regex pattern to match dataset names
        
    Returns
    -------
    list
        List of dataset names that match the criteria
        
    Examples
    --------
    >>> # Find datasets with more than 1000 rows
    >>> find_datasets(criteria=lambda df: len(df) > 1000)
    
    >>> # Find datasets with columns that look like OHLC data
    >>> find_datasets(column_pattern='open|high|low|close|volume')
    
    >>> # Find datasets with 'forex' in their name
    >>> find_datasets(name_pattern='forex')
    """
    import re
    
    matches = []
    
    # Compile regex patterns if provided
    col_regex = re.compile(column_pattern, re.IGNORECASE) if column_pattern else None
    name_regex = re.compile(name_pattern, re.IGNORECASE) if name_pattern else None
    
    # Iterate through all datasets
    for name in DS_MANAGER.names():
        df = DS_MANAGER[name]
        
        # Check name pattern
        if name_regex and not name_regex.search(name):
            continue
        
        # Check column pattern
        if col_regex and not any(col_regex.search(col) for col in df.columns):
            continue
            
        # Check criteria function
        if criteria and not criteria(df):
            continue
            
        # If we got here, the dataset matches all provided criteria
        matches.append(name)
    
    # Print results
    if matches:
        print(f"Found {len(matches)} matching datasets:")
        for i, name in enumerate(matches):
            df = DS_MANAGER[name]
            print(f"{i}: {name} - Shape: {df.shape}")
    else:
        print("No matching datasets found.")
    
    return matches


# -------------------------------------------------------------------------
# Model for QTableView – minimal Pandas → Qt adapter
# -------------------------------------------------------------------------


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
        self._mpl_toolbar = None # Explicitly initialize to None

        # Matplotlib figure -----------------------------------------------------------------
        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)

        self._ax = self._fig.add_subplot(111)

        # State
        self._chart_mode: str = "Line"  # or "Candlestick"
        
        # Color mapping for columns
        self._color_map = {}
        self._default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Default columns: Close if present else first numeric
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if "Close" in df.columns:
            self._columns = ["Close"]
        elif numeric_cols:
            self._columns = [numeric_cols[0]]
        else:
            self._columns = []

        # Drawing tools integration
        self.drawing_manager = None
        self.drawing_renderer = None
        self.coordinate_transformer = None
        self.drawing_enabled = HAS_DRAWING_TOOLS
        
        if self.drawing_enabled:
            self._setup_drawing_tools()

        self._draw()

    @property
    def mpl_toolbar(self):
        """Get the matplotlib toolbar reference."""
        return self._mpl_toolbar
    
    @mpl_toolbar.setter
    def mpl_toolbar(self, value):
        """Set the matplotlib toolbar reference."""
        self._mpl_toolbar = value

    def _setup_drawing_tools(self):
        """Initialize drawing tools if available."""
        try:
            self.drawing_manager = DrawingManager()
            self.coordinate_transformer = CoordinateTransformer(self._ax, self._df)
            self.drawing_renderer = DrawingRenderer(self._ax, self.coordinate_transformer)
            
            # Connect drawing manager signals
            self.drawing_manager.drawing_added.connect(self._on_drawing_added)
            self.drawing_manager.drawing_removed.connect(self._on_drawing_removed)
            self.drawing_manager.drawing_updated.connect(self._on_drawing_updated)
            
            # Enable mouse events for drawing
            self.mpl_connect('button_press_event', self._on_mouse_press)
            self.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.mpl_connect('button_release_event', self._on_mouse_release)
            self.mpl_connect('key_press_event', self._on_key_press)
            
            # Set focus policy to receive key events
            self.setFocusPolicy(Qt.StrongFocus)
            
            # Initialize cursor state
            self._default_cursor = self.cursor()
            
            # Track mouse state for proper click-and-drag behavior
            self._mouse_pressed = False
            self._mouse_moved_during_press = False
            
            # Flag to prevent chart redraws during drawing operations
            self._drawing_in_progress = False
            
            # Store current view limits for preserving zoom
            self._saved_xlim = None
            self._saved_ylim = None
            
        except Exception as e:
            print(f"Failed to setup drawing tools: {e}")
            self.drawing_enabled = False

    def _save_view_limits(self):
        """Save current axis view limits."""
        if self._ax:
            self._saved_xlim = self._ax.get_xlim()
            self._saved_ylim = self._ax.get_ylim()
            
    def _restore_view_limits(self):
        """Restore saved axis view limits."""
        if self._ax and self._saved_xlim and self._saved_ylim:
            self._ax.set_xlim(self._saved_xlim)
            self._ax.set_ylim(self._saved_ylim)

    def set_drawing_tool(self, tool_type):
        """Set the current drawing tool and update cursor."""
        if not self.drawing_enabled or not self.drawing_manager:
            return
            
        self.drawing_manager.set_current_tool(tool_type)
        self._update_cursor_for_tool(tool_type)
        
    def _update_cursor_for_tool(self, tool_type):
        """Update the cursor based on the selected tool."""
        if tool_type == DrawingToolType.NONE:
            # Default cursor for selection/navigation
            self.setCursor(Qt.PointingHandCursor)
        else:
            # Crosshair cursor for drawing tools
            self.setCursor(Qt.CrossCursor)

    def _on_drawing_added(self, drawing):
        """Handle new drawing added."""
        if self.drawing_renderer:
            self.drawing_renderer.render_drawing(drawing)
            # Use canvas draw directly to avoid triggering complete chart redraw
            self.figure.canvas.draw_idle() # Changed to draw_idle for consistency

    def _on_drawing_removed(self, drawing_id):
        """Handle drawing removed."""
        if self.drawing_renderer:
            self.drawing_renderer.remove_drawing(drawing_id)
            # Use canvas draw directly to avoid triggering complete chart redraw
            self.figure.canvas.draw_idle() # Changed to draw_idle for consistency

    def _on_drawing_updated(self, drawing):
        """Handle drawing updated (e.g., moved, style changed, selected)."""
        if self.drawing_renderer:
            # --- Preserve zoom: capture limits and autoscale state ---
            current_xlim = self._ax.get_xlim()
            current_ylim = self._ax.get_ylim()
            was_autoscale = self._ax.get_autoscale_on()
            self._ax.set_autoscale_on(False)
            
            # Clear any existing resize handles
            self.drawing_renderer.clear_resize_handles()
            
            # Re-render the specific drawing
            self.drawing_renderer.render_drawing(drawing)
            
            # If the drawing is selected, render resize handles
            if drawing.is_selected:
                self.drawing_renderer.render_resize_handles(drawing, self.coordinate_transformer)
            
            # --- Restore limits and autoscale ---
            self._ax.set_xlim(current_xlim)
            self._ax.set_ylim(current_ylim)
            self._ax.set_autoscale_on(was_autoscale)
            self.figure.canvas.draw_idle()

    def _on_mouse_press(self, event):
        """Handle mouse press events for drawing."""
        mpl_tool_active = False
        current_mode = None # Initialize current_mode
        if self._mpl_toolbar:
            # Check if it's actually a NavigationToolbar instance
            try:
                # NavigationToolbar should have a 'mode' attribute
                current_mode = self._mpl_toolbar.mode
                if current_mode and current_mode != '': # Check if current_mode is not None and not an empty string
                    mpl_tool_active = True
            except AttributeError:
                # If mode doesn't exist, it's not a proper NavigationToolbar
                print(f"WARNING: ChartWidget._on_mouse_press: mpl_toolbar doesn't have 'mode' attribute. Type: {type(self._mpl_toolbar)}")
                current_mode = None

        # If MPL nav tool is active, do nothing from our side and cancel any pending drawing.
        if mpl_tool_active and self.drawing_manager: # Use the new mpl_tool_active flag
            if self.drawing_manager.drawing_in_progress:
                self.drawing_manager.cancel_drawing()
                # Also reset our internal drawing flags if any were set
                self._mouse_pressed = False
                self._drawing_in_progress = False
                if hasattr(self, '_original_xlim'): # Ensure autoscale is re-enabled
                    self._ax.set_autoscale_on(True)
                self.figure.canvas.draw_idle()
            return # Let MPL handle it

        if not self.drawing_enabled or not self.drawing_manager:
            return
            
        if event.inaxes != self._ax:
            return

        # Handle selection tool logic first
        current_tool_in_manager = self.drawing_manager.current_tool

        if current_tool_in_manager == DrawingToolType.NONE: # NONE is effectively our SELECT tool
            
            # First check if we're clicking on a resize handle
            handle_index = self.drawing_manager.get_resize_handle_at(event.xdata, event.ydata, self.coordinate_transformer, self._ax)
            
            if handle_index is not None and event.button == 1:
                # Start resizing
                self.drawing_manager.start_resizing_selected_drawing(handle_index, event.xdata, event.ydata, self.coordinate_transformer)
                self._mouse_pressed = True
                self._mouse_moved_during_press = False
                # Keep axes fixed during resize
                self._original_xlim = self._ax.get_xlim()
                self._original_ylim = self._ax.get_ylim()
                self._ax.set_autoscale_on(False)
                return True
            
            # If not on a handle, check for selection
            clicked_something = self.drawing_manager.select_drawing_at(event.xdata, event.ydata, self.coordinate_transformer, self.drawing_renderer.rendered_objects, self._ax)
            selected_drawing = self.drawing_manager.get_selected_drawing()

            if event.button == 1: # Left click
                if selected_drawing: # If an object was selected, start dragging it
                    self.drawing_manager.start_dragging_selected_drawing(event.xdata, event.ydata, self.coordinate_transformer)
                    self._mouse_pressed = True # Ensure _on_mouse_move knows button is down for drag
                    self._mouse_moved_during_press = False # Initialize for drag
                    # Keep axes fixed during drag
                    self._original_xlim = self._ax.get_xlim()
                    self._original_ylim = self._ax.get_ylim()
                    self._ax.set_autoscale_on(False)
                else: # Clicked on empty space, deselect if anything was selected
                    self.drawing_manager._set_selection(None) # Explicitly deselect
            
            # Potentially update style panel here if a drawing is selected or deselected
            # This should be connected to drawing_manager.selection_changed signal by the toolbar
            self.figure.canvas.draw_idle() # Redraw to show selection changes
            return True # Consume event if select tool is active
            
        # This check was a duplicate from the original template, ensure it's properly part of drawing logic now
        # if self.drawing_manager.current_tool == DrawingToolType.NONE: 
        #     return
            
        # Only handle left mouse button for drawing tools
        if event.button != 1:
            return
        
        # Store original axis limits to prevent matplotlib from changing them during drawing
        self._original_xlim = self._ax.get_xlim()
        self._original_ylim = self._ax.get_ylim()
        # Disable autoscaling during drawing
        self._ax.set_autoscale_on(False)
        
        # Set mouse state and drawing in progress flag
        self._mouse_pressed = True
        self._mouse_moved_during_press = False
        self._drawing_in_progress = True
        
        # For text tool, immediately show input dialog
        if self.drawing_manager.current_tool == DrawingToolType.TEXT:
            # Use QInputDialog for simplicity
            text, ok = QInputDialog.getText(self, "Enter Text", "Text:")
            if ok and text:
                # Start and finish text drawing immediately
                drawing_id = self.drawing_manager.start_drawing(
                    event.xdata, event.ydata, self.coordinate_transformer
                )
                if drawing_id:
                    self.drawing_manager.finish_drawing(text)
            self._mouse_pressed = False
            self._drawing_in_progress = False
            # Restore axis limits and re-enable autoscaling
            self._ax.set_xlim(self._original_xlim)
            self._ax.set_ylim(self._original_ylim)
            self._ax.set_autoscale_on(True)
            self.figure.canvas.draw_idle() # Use draw_idle for efficiency
            return True
        
        # For other tools, start drawing but don't finish until mouse release with movement
        drawing_id = self.drawing_manager.start_drawing(
            event.xdata, event.ydata, self.coordinate_transformer
        )
        
        # Consume the event to prevent matplotlib from processing it
        return True

    def _on_mouse_move(self, event):
        mpl_tool_active = False
        current_mode = None # Initialize current_mode
        if self._mpl_toolbar:
            # Check if it's actually a NavigationToolbar instance
            try:
                # NavigationToolbar should have a 'mode' attribute
                current_mode = self._mpl_toolbar.mode
                if current_mode and current_mode != '': # Check if current_mode is not None and not an empty string
                    mpl_tool_active = True
            except AttributeError:
                # If mode doesn't exist, it's not a proper NavigationToolbar
                # Don't spam the console with warnings in mouse move
                current_mode = None
                
        # If MPL nav tool is active, do nothing from our side.
        if mpl_tool_active and self.drawing_manager: # Use the new mpl_tool_active flag
            # We might still want our cursor updates if our select tool is also selected
            # but for now, let MPL have full control if its tool is active.
            return # Let MPL handle it

        if not self.drawing_enabled or not self.drawing_manager or not self._ax:
            return

        if event.xdata is None or event.ydata is None: # Outside axes
            # If mouse is outside, but a drawing was in progress, might want to update cursor to default
            if not self._mouse_pressed and self.drawing_manager.current_tool != DrawingToolType.NONE:
                self.setCursor(self._default_cursor) # Or specific application default
            return

        # If Select tool is active and no button is pressed, ensure pointing hand cursor
        if self.drawing_manager.current_tool == DrawingToolType.NONE and not self._mouse_pressed:
            # Check if we're over a resize handle
            handle_index = self.drawing_manager.get_resize_handle_at(event.xdata, event.ydata, self.coordinate_transformer, self._ax)
            if handle_index is not None:
                # Show resize cursor based on handle position
                if self.drawing_manager.selected_drawing_id and self.drawing_manager.selected_drawing_id in self.drawing_manager.drawings:
                    drawing = self.drawing_manager.drawings[self.drawing_manager.selected_drawing_id]
                    if drawing.tool_type == DrawingToolType.LINE:
                        self.setCursor(Qt.CrossCursor)  # Cross cursor for line endpoints
                    else:
                        # Different cursors for different handles on rectangles/ovals
                        if handle_index in [0, 4]:  # Top-left, bottom-right
                            self.setCursor(Qt.SizeFDiagCursor)
                        elif handle_index in [2, 6]:  # Top-right, bottom-left
                            self.setCursor(Qt.SizeBDiagCursor)
                        elif handle_index in [1, 5]:  # Top, bottom
                            self.setCursor(Qt.SizeVerCursor)
                        elif handle_index in [3, 7]:  # Left, right
                            self.setCursor(Qt.SizeHorCursor)
            elif self.cursor().shape() != Qt.PointingHandCursor:
                self.setCursor(Qt.PointingHandCursor)
        elif self.drawing_manager.current_tool != DrawingToolType.NONE and not self._mouse_pressed:
            # For other drawing tools, ensure cross cursor when hovering and not drawing
            if self.cursor().shape() != Qt.CrossCursor:
                self.setCursor(Qt.CrossCursor)

        # Drawing logic (if mouse button is pressed)
        if self._mouse_pressed and self.drawing_manager.drawing_in_progress:
            self._mouse_moved_during_press = True  # Set this flag to indicate movement occurred
            self.drawing_manager.update_drawing(
                event.xdata, event.ydata, self.coordinate_transformer
            )
            # Re-render the active drawing
            if self.drawing_manager.active_drawing:
                self.drawing_renderer.render_drawing(self.drawing_manager.active_drawing)
                # Ensure axis limits are maintained (they should be if autoscale is off)
                self._ax.set_xlim(self._original_xlim)
                self._ax.set_ylim(self._original_ylim)
                # Use canvas draw directly to avoid triggering chart redraw
                self.figure.canvas.draw_idle() # Use draw_idle
            
            # Consume the event to prevent matplotlib from processing it
            return True
            
        # Handle dragging for selection tool
        if self._mouse_pressed and self.drawing_manager.current_tool == DrawingToolType.NONE and self.drawing_manager.is_dragging_selection():
            self._mouse_moved_during_press = True
            self.drawing_manager.drag_selected_drawing(event.xdata, event.ydata, self.coordinate_transformer)
            # Ensure axis limits are maintained during drag
            if hasattr(self, '_original_xlim'):
                self._ax.set_xlim(self._original_xlim)
                self._ax.set_ylim(self._original_ylim)
            self.figure.canvas.draw_idle()
            return True

        # Handle resizing for selection tool
        if self._mouse_pressed and self.drawing_manager.current_tool == DrawingToolType.NONE and self.drawing_manager.is_resizing_selection():
            self._mouse_moved_during_press = True
            self.drawing_manager.resize_selected_drawing(event.xdata, event.ydata, self.coordinate_transformer)
            # Ensure axis limits are maintained during resize
            if hasattr(self, '_original_xlim'):
                self._ax.set_xlim(self._original_xlim)
                self._ax.set_ylim(self._original_ylim)
            self.figure.canvas.draw_idle()
            return True

    def _on_mouse_release(self, event):
        if not self.drawing_enabled or not self.drawing_manager:
            return

        # Store if a drawing operation was truly active before we reset flags
        was_drawing_in_progress_chart_widget = self._drawing_in_progress
        was_drawing_in_progress_manager = self.drawing_manager.drawing_in_progress
        
        # Handle end of resizing for selection tool
        if event.button == 1 and self.drawing_manager.current_tool == DrawingToolType.NONE and self.drawing_manager.is_resizing_selection():
            self.drawing_manager.finish_resizing_selected_drawing()
            self._mouse_pressed = False
            self._mouse_moved_during_press = False
            self._ax.set_autoscale_on(True) # Enable, but don't force autoscale view yet
            if hasattr(self, '_original_xlim'): del self._original_xlim
            if hasattr(self, '_original_ylim'): del self._original_ylim
            self.figure.canvas.draw_idle()
            return True

        # Handle end of dragging for selection tool
        if event.button == 1 and self.drawing_manager.current_tool == DrawingToolType.NONE and self.drawing_manager.is_dragging_selection():
            self.drawing_manager.finish_dragging_selected_drawing()
            self._mouse_pressed = False
            self._mouse_moved_during_press = False
            self._ax.set_autoscale_on(True) # Enable, but don't force autoscale view yet
            if hasattr(self, '_original_xlim'): del self._original_xlim
            if hasattr(self, '_original_ylim'): del self._original_ylim
            self.figure.canvas.draw_idle()
            return True
            
        if event.inaxes != self._ax:
            self._mouse_pressed = False
            self._mouse_moved_during_press = False
            self._drawing_in_progress = False # Reset chart widget flag
            if self.drawing_manager.drawing_in_progress: # If manager was still drawing
                 self.drawing_manager.cancel_drawing() # Cancel it
            if hasattr(self, '_original_xlim'): # If we had stored limits
                self._ax.set_xlim(self._original_xlim)
                self._ax.set_ylim(self._original_ylim)
                # Autoscale should be re-enabled only after restoring limits
                self._ax.set_autoscale_on(True) 
                del self._original_xlim
                del self._original_ylim
            else: # No stored limits, just ensure autoscale is on
                 self._ax.set_autoscale_on(True)
            self.figure.canvas.draw_idle()
            return

        if event.button != 1:
            return
            
        # At this point, it's a left-click release inside the axes
        
        acted_on_drawing = False
        # Check based on chart_widget's flag, as manager's flag might be reset by start_drawing again if no move
        if was_drawing_in_progress_chart_widget and self._mouse_moved_during_press: 
            if self.drawing_manager.current_tool != DrawingToolType.TEXT: # Text is handled on press
                self.drawing_manager.finish_drawing() 
                acted_on_drawing = True
                # Zoom should be preserved because autoscale was off during press/move
                # and we are not calling autoscale_view() here.
                # Limits stored in _original_xlim/ylim were for *during* the draw.
                # We want to keep the *current* view after the draw.
                if hasattr(self, '_original_xlim'): del self._original_xlim
                if hasattr(self, '_original_ylim'): del self._original_ylim
                
        elif was_drawing_in_progress_chart_widget: # Click without move while a drawing tool was active (and widget thought it was drawing)
            if self.drawing_manager.drawing_in_progress: # Check manager too, as it might have been a very quick click-drag-click
                self.drawing_manager.cancel_drawing()
                acted_on_drawing = True
                # Restore axis limits from before the click
                if hasattr(self, '_original_xlim'):
                    self._ax.set_xlim(self._original_xlim)
                    self._ax.set_ylim(self._original_ylim)
                    del self._original_xlim
                    del self._original_ylim
            # else: manager doesn't think it's drawing, so maybe nothing to cancel that would affect view
        
        # General cleanup
        self._mouse_pressed = False
        self._drawing_in_progress = False # Reset ChartWidget's flag
        self._mouse_moved_during_press = False

        # Re-enable autoscaling for future MPL actions, but current view should persist
        if acted_on_drawing:
            # Preserve current view after drawing
            current_xlim = self._ax.get_xlim()
            current_ylim = self._ax.get_ylim()
            self._ax.set_autoscale_on(True)
            self._ax.set_xlim(current_xlim)
            self._ax.set_ylim(current_ylim)
            self.figure.canvas.draw_idle()
        else:
            self._ax.set_autoscale_on(True)

        return True

    def _on_key_press(self, event):
        """Handle key press events for drawing tools."""
        if not self.drawing_enabled or not self.drawing_manager:
            return
            
        # Delete selected drawing
        if event.key == 'delete':
            selected = self.drawing_manager.get_selected_drawing()
            if selected:
                self.drawing_manager.remove_drawing(selected.id)
                self.draw_idle()
                return
            
        # Tool shortcuts
        if event.key == 'escape':
            if self.drawing_manager.drawing_in_progress:
                self.drawing_manager.cancel_drawing()
                self.draw_idle()
            else:
                self.drawing_manager.set_current_tool(DrawingToolType.NONE)
        elif event.key == 'l':
            self.drawing_manager.set_current_tool(DrawingToolType.LINE)
        elif event.key == 'r':
            self.drawing_manager.set_current_tool(DrawingToolType.RECTANGLE)
        elif event.key == 'o':
            self.drawing_manager.set_current_tool(DrawingToolType.OVAL)
        elif event.key == 't':
            self.drawing_manager.set_current_tool(DrawingToolType.TEXT)

    def update_drawing_data(self, new_df: pd.DataFrame):
        """Update the chart data and refresh drawings."""
        self._df = new_df
        if self.drawing_enabled and self.coordinate_transformer:
            # Update coordinate transformer with new data
            self.coordinate_transformer.df = new_df
            # Re-render all drawings with new coordinates
            if self.drawing_renderer:
                drawings = self.drawing_manager.get_all_drawings()
                self.drawing_renderer.refresh_all(drawings)

    def get_color(self, column, idx=None):
        """Get color for a column, create if doesn't exist yet."""
        if column not in self._color_map:
            # Assign default color based on index 
            if idx is not None:
                color_idx = idx % len(self._default_colors)
            else:
                color_idx = len(self._color_map) % len(self._default_colors)
            self._color_map[column] = self._default_colors[color_idx]
        return self._color_map[column]
        
    def set_color(self, column, color):
        """Set the color for a specific column."""
        self._color_map[column] = color
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
        """Draw or update the chart based on current settings."""
        # Skip redraw if drawing operation is in progress to prevent clearing drawings
        if hasattr(self, '_drawing_in_progress') and self._drawing_in_progress:
            return
            
        # Clear existing axes
        self._fig.clear()
        
        # Identify columns with significantly different scales
        main_cols = []
        secondary_cols = []
        
        if self._chart_mode == "Candlestick" and self._has_ohlc():
            # For candlestick mode, the main plot is always OHLC
            ohlc_cols = self._find_ohlc_cols()
            cc = ohlc_cols[3]  # close column
            baseline_mean = self._df[cc].mean()
            
            # Identify other columns that should go on a different scale
            for col in self._columns:
                if col not in ohlc_cols and col in self._df.columns:
                    if (self._df[col].mean() > baseline_mean * 2 or 
                        self._df[col].mean() < baseline_mean / 2 or
                        self._df[col].std() / self._df[col].mean() > 
                        self._df[cc].std() / baseline_mean * 5):
                        secondary_cols.append(col)
                    else:
                        # If scale is similar to OHLC, add to main chart
                        main_cols.append(col)
        else:
            # For line chart mode, base scaling on first selected column
            if self._columns and self._columns[0] in self._df.columns:
                baseline_col = self._columns[0]
                baseline_mean = self._df[baseline_col].mean()
                baseline_std = self._df[baseline_col].std()
                
                main_cols.append(baseline_col)
                
                # Categorize other columns by their scale relative to the first column
                for col in self._columns[1:]:
                    if col in self._df.columns:
                        if (self._df[col].mean() > baseline_mean * 2 or 
                            self._df[col].mean() < baseline_mean / 2 or
                            self._df[col].std() / self._df[col].mean() > 
                            baseline_std / baseline_mean * 5):
                            secondary_cols.append(col)
                        else:
                            main_cols.append(col)
        
        # Create one or two subplots based on whether we have secondary columns
        if secondary_cols:
            # Create a figure with a 2:1 height ratio between main and secondary plots
            gs = self._fig.add_gridspec(3, 1)  # 3 rows, 1 column
            self._ax = self._fig.add_subplot(gs[0:2, 0])  # Main plot takes 2/3 of height
            self._ax2 = self._fig.add_subplot(gs[2, 0], sharex=self._ax)  # Secondary takes 1/3
        else:
            # Only one plot needed
            self._ax = self._fig.add_subplot(1, 1, 1)
            self._ax2 = None
        
        # Update coordinate transformer AND drawing_renderer with new axes (BEFORE drawing chart)
        if self.drawing_enabled:
            if self.coordinate_transformer:
                self.coordinate_transformer.ax = self._ax
                self.coordinate_transformer.df = self._df # Ensure df is also current
            if self.drawing_renderer: # THIS IS THE NEWLY ADDED IMPORTANT LINE
                self.drawing_renderer.ax = self._ax
        
        # Draw the appropriate chart type
        if self._chart_mode == "Candlestick" and self._has_ohlc():
            # First plot candlesticks
            self._plot_candles()
            
            # Now overlay any main columns on the candlestick chart
            for i, col in enumerate(main_cols):
                color = self.get_color(col, i)
                self._ax.plot(range(len(self._df)), self._df[col].values, 
                             label=col, linewidth=1.5, color=color)
            
            # Add legend if we have overlay indicators
            if main_cols:
                self._ax.legend(loc="upper left", fontsize="small")
                
            # Now plot any secondary columns in the second subplot
            if self._ax2 and secondary_cols:
                for i, col in enumerate(secondary_cols):
                    color = self.get_color(col, i + len(main_cols))
                    self._ax2.plot(range(len(self._df)), self._df[col].values, 
                                  label=col, linewidth=1.5, color=color)
                self._ax2.legend(loc="upper left", fontsize="small")
                self._ax2.set_title(", ".join(secondary_cols))
                self._ax2.grid(True, linestyle=":", alpha=0.4)
        else:
            # Line chart mode
            if main_cols:
                for i, col in enumerate(main_cols):
                    color = self.get_color(col, i)
                    self._ax.plot(self._df[col].values, label=col, color=color)
                if len(main_cols) > 1:
                    self._ax.legend(loc="upper left", fontsize="small")
                self._ax.set_title(", ".join(main_cols))
            
            # Plot secondary columns in second subplot
            if self._ax2 and secondary_cols:
                for i, col in enumerate(secondary_cols):
                    color = self.get_color(col, i + len(main_cols))
                    self._ax2.plot(self._df[col].values, label=col, color=color)
                self._ax2.legend(loc="upper left", fontsize="small")
                self._ax2.set_title(", ".join(secondary_cols))
                self._ax2.grid(True, linestyle=":", alpha=0.4)
        
        # Common settings for main axis
        self._ax.grid(True, linestyle=":", alpha=0.4)
        
        # Update coordinate transformer AFTER chart is drawn and axes limits are set
        if self.drawing_enabled and self.coordinate_transformer:
            self.coordinate_transformer.ax = self._ax
            # self.coordinate_transformer.df = self._df # df is already set above
            # print(f"Updated coordinate transformer - axes limits: xlim={self._ax.get_xlim()}, ylim={self._ax.get_ylim()}") # Keep this commented
        
        # Overlays (only on the main plot) -----------------------------------------------
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
        
        # Re-render drawings if drawing tools are enabled
        if self.drawing_enabled and self.drawing_renderer:
            # Ensure renderer has the latest axes before refreshing
            self.drawing_renderer.ax = self._ax 
            drawings = self.drawing_manager.get_all_drawings()
            self.drawing_renderer.refresh_all(drawings)
        
        # Adjust layout
        self._fig.tight_layout()
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
            # Try modern mplfinance first
            import mplfinance as mpf
            
            ohlc_cols = self._find_ohlc_cols()
            if ohlc_cols is None:
                raise RuntimeError("No OHLC columns found")

            # Prepare data for modern mplfinance
            ohlc_data = self._df.loc[:, ohlc_cols].copy()
            ohlc_data.columns = ['Open', 'High', 'Low', 'Close']  # Ensure standard column names
            
            # Create the plot using modern mplfinance
            # Since we're working within an existing matplotlib axis, we need to use a different approach
            
            # Fall back to manual candlestick drawing
            self._plot_candles_manual()
            
        except ImportError:
            # Fall back to manual candlestick drawing if mplfinance is not available
            self._plot_candles_manual()
        except Exception as e:
            print(f"Error with modern mplfinance: {e}")
            # Fall back to manual candlestick drawing
            self._plot_candles_manual()
            
    def _plot_candles_manual(self):
        """Manual candlestick plotting using matplotlib rectangles and lines."""
        try:
            import matplotlib.patches as patches
            import matplotlib.dates as mdates
            
            ohlc_cols = self._find_ohlc_cols()
            if ohlc_cols is None:
                raise RuntimeError("No OHLC columns found")

            ohlc = self._df.loc[:, ohlc_cols].copy()
            oc, hc, lc, cc = ohlc_cols

            # Use sequential x-values to eliminate weekend gaps
            x_values = list(range(len(ohlc)))
            use_datetime = isinstance(ohlc.index, pd.DatetimeIndex)

            # Calculate candle width (fixed width for sequential plotting)
            width = 0.6

            # Define colors that work well on both light and dark themes
            bullish_color = '#00C851'  # Bright green
            bearish_color = '#FF4444'  # Bright red
            wick_color = '#CCCCCC'     # Light gray for wicks
            
            # Check if we're on a dark theme by looking at the figure background
            fig_bg = self._fig.get_facecolor()
            if isinstance(fig_bg, tuple) and len(fig_bg) >= 3:
                # If background is dark (sum of RGB < 1.5), use lighter wick color
                if sum(fig_bg[:3]) < 1.5:
                    wick_color = '#DDDDDD'  # Lighter gray for dark themes
                else:
                    wick_color = '#666666'  # Darker gray for light themes

            # First pass: Draw all wicks (behind candle bodies)
            for i, (idx, row) in enumerate(ohlc.iterrows()):
                x = x_values[i]
                high_price = float(row[hc])
                low_price = float(row[lc])
                open_price = float(row[oc])
                close_price = float(row[cc])
                
                # Only draw the parts of wicks that extend beyond the body
                body_top = max(open_price, close_price)
                body_bottom = min(open_price, close_price)
                
                # Draw upper wick (from body top to high)
                if high_price > body_top:
                    self._ax.plot([x, x], [body_top, high_price], 
                                color=wick_color, linewidth=1.5, alpha=0.8, zorder=1)
                
                # Draw lower wick (from low to body bottom)
                if low_price < body_bottom:
                    self._ax.plot([x, x], [low_price, body_bottom], 
                                color=wick_color, linewidth=1.5, alpha=0.8, zorder=1)

            # Second pass: Draw all candle bodies (on top of wicks)
            for i, (idx, row) in enumerate(ohlc.iterrows()):
                x = x_values[i]
                open_price = float(row[oc])
                high_price = float(row[hc])
                low_price = float(row[lc])
                close_price = float(row[cc])
                
                # Determine if bullish or bearish
                is_bullish = close_price >= open_price
                color = bullish_color if is_bullish else bearish_color
                
                # Draw the open-close rectangle (body)
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                if body_height > 0:
                    # Draw filled rectangle for the body WITHOUT outline and fully opaque
                    rect = patches.Rectangle(
                        (x - width/2, body_bottom), 
                        width, 
                        body_height,
                        facecolor=color, 
                        edgecolor='none',  # Remove outline completely
                        linewidth=0,       # No border
                        alpha=1.0,         # Fully opaque to hide wicks
                        zorder=2  # Draw bodies on top of wicks
                    )
                    self._ax.add_patch(rect)
                else:
                    # Doji (open == close) - draw a horizontal line
                    self._ax.plot([x - width/2, x + width/2], [open_price, open_price], 
                                color=wick_color, linewidth=2.5, alpha=0.9, zorder=2)

            # Format x-axis with better date/time display
            if use_datetime:
                # Create custom tick positions and labels
                num_ticks = min(10, len(ohlc))  # Limit number of ticks
                if num_ticks > 1:
                    tick_indices = [int(i * (len(ohlc) - 1) / (num_ticks - 1)) for i in range(num_ticks)]
                else:
                    tick_indices = [0]
                
                tick_positions = [x_values[i] for i in tick_indices]
                tick_labels = []
                
                for i in tick_indices:
                    timestamp = ohlc.index[i]
                    # Format as date + time for better readability
                    if hasattr(timestamp, 'strftime'):
                        # Show both date and time
                        label = timestamp.strftime('%m/%d\n%H:%M')
                    else:
                        label = str(timestamp)
                    tick_labels.append(label)
                
                self._ax.set_xticks(tick_positions)
                self._ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=8)
            else:
                # For non-datetime data, use simple numeric labels
                num_ticks = min(10, len(x_values))
                if num_ticks > 1:
                    tick_positions = [int(i * (len(x_values) - 1) / (num_ticks - 1)) for i in range(num_ticks)]
                    self._ax.set_xticks(tick_positions)

            # Set y-axis limits with padding
            all_prices = []
            for _, row in ohlc.iterrows():
                all_prices.extend([row[oc], row[hc], row[lc], row[cc]])
            
            if all_prices:
                min_price, max_price = min(all_prices), max(all_prices)
                price_range = max_price - min_price
                padding = price_range * 0.05 if price_range > 0 else 0.01
                self._ax.set_ylim(min_price - padding, max_price + padding)

            self._ax.set_title(f"OHLC Candlestick - {self.dataset_name or 'Live Chart'}")
            
            # Grid with theme-appropriate alpha
            grid_alpha = 0.2 if sum(fig_bg[:3]) < 1.5 else 0.3
            self._ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)

        except Exception as e:
            print(f"Error plotting manual candlesticks: {e}")
            # Final fallback to Close line
            self._ax.plot(self._df["Close"].values, label="Close", color="blue")
            self._ax.legend(loc="upper left", fontsize="small")
            self._ax.set_title("Close Price (candlestick error)")


class Workspace(QWidget):
    """A workspace shows one dataset in both chart & table form."""

    def __init__(self, dataset_name: str, parent: QWidget | None = None):
        super().__init__(parent)

        self.dataset_name = dataset_name

        df = DS_MANAGER[dataset_name]

        # Import necessary Qt components to ensure they're in local scope
        from PySide6.QtCore import Qt, QSize, QEvent, QObject
        from PySide6.QtGui import QColor, QPalette, QPixmap, QPainter
        from PySide6.QtWidgets import (
            QLabel, QComboBox, QCheckBox, QScrollArea, QSplitter, QToolButton,
            QHBoxLayout, QVBoxLayout, QSizePolicy, QFrame, QMenu, QApplication,
            QColorDialog, QWidgetAction
        )

        # ------------------------------------------------------------------
        # Main layout setup
        # ------------------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create main horizontal splitter for chart/table and drawing toolbar
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create a splitter for chart/table
        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter  # Save reference for later
        
        # Create chart first so we can reference it
        chart = ChartWidget(df=df, dataset_name=dataset_name, parent=splitter)
        splitter.addWidget(chart)
        self.chart = chart  # Save reference for later
        
        # Create table view (only once!)
        self.table = QTableView()
        model = _create_pandas_model(df)
        self.table.setModel(model)
        self.table.setSortingEnabled(True)
        splitter.addWidget(self.table)
        # Table should be visible to the splitter, but collapsed by default
        self.table.setStyleSheet("")
        
        # Add chart/table splitter to main splitter
        main_splitter.addWidget(splitter)
        
        # ------------------------------------------------------------------
        # Compact ribbon-style toolbar (create first to get nav_toolbar reference)
        # ------------------------------------------------------------------
        toolbar = QFrame(self)
        toolbar.setFrameShape(QFrame.StyledPanel)
        toolbar.setMaximumHeight(36)  # Keep toolbar height minimal
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)
        toolbar_layout.setSpacing(8)

        # Chart mode selector
        toolbar_layout.addWidget(QLabel("Chart:"))
        mode_combo = QComboBox()
        mode_combo.addItems(["Line", "Candlestick"])
        mode_combo.setMaximumWidth(100)
        toolbar_layout.addWidget(mode_combo)

        # Columns dropdown button
        columns_btn = QToolButton()
        columns_btn.setText("Columns")
        columns_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar_layout.addWidget(columns_btn)

        # Create a persistent dropdown menu for columns using QMenu subclass
        class PersistentMenu(QMenu):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
                self.setAttribute(Qt.WA_TranslucentBackground)
            
            def mouseReleaseEvent(self, event):
                # Don't close the menu when an action is clicked - only when clicked outside
                action = self.actionAt(event.pos())
                if action and action.isEnabled() and not action.isSeparator():
                    action.trigger()
                    event.accept()
                    return
                super().mouseReleaseEvent(event)
        
        columns_menu = PersistentMenu(columns_btn)
        columns_btn.setMenu(columns_menu)
        
        # Dictionary to store column checkboxes and colors
        column_checkboxes = {}
        column_colors = {}
        
        # Create custom widget actions for better visibility of checkboxes
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Create a QWidgetAction for more control over appearance
                widget_action = QWidgetAction(columns_menu)
                
                # Create widget to hold checkbox and label
                action_widget = QWidget()
                action_layout = QHBoxLayout(action_widget)
                action_layout.setContentsMargins(4, 2, 4, 2)
                action_layout.setSpacing(6)
                
                # Add checkbox with clear visibility
                checkbox = QCheckBox()
                # Pre-check Close or first numeric column
                if col in ("Close",) or col == df.columns[0]:
                    checkbox.setChecked(True)
                else:
                    checkbox.setChecked(False)
                action_layout.addWidget(checkbox)
                
                # Get color for this column
                color = chart.get_color(col)
                column_colors[col] = color
                
                # Create colored square icon
                def create_color_icon(color_str, size=16):
                    pixmap = QPixmap(size, size)
                    pixmap.fill(Qt.transparent)
                    painter = QPainter(pixmap)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(color_str))
                    painter.drawRect(2, 2, size-4, size-4)
                    painter.end()
                    return pixmap
                
                # Create label with colored icon
                label = QLabel(col)
                label.setStyleSheet("font-weight: normal;")
                color_pixmap = create_color_icon(color)
                color_label = QLabel()
                color_label.setPixmap(color_pixmap)
                color_label.setFixedSize(16, 16)
                
                # Add color square and text label
                action_layout.addWidget(color_label)
                action_layout.addWidget(label)
                action_layout.addStretch()
                
                # Set the widget as the action's widget
                widget_action.setDefaultWidget(action_widget)
                columns_menu.addAction(widget_action)
                
                # Store references for later
                column_checkboxes[col] = {"action": widget_action, "checkbox": checkbox, "color_label": color_label}
                
                # Connect checkbox changes to update the chart
                def create_checkbox_handler(column):
                    def handle_checkbox():
                        _update_columns()
                    return handle_checkbox
                
                checkbox.stateChanged.connect(create_checkbox_handler(col))
                
                # Right-click handler for color label
                def create_color_picker_handler(column, color_label_widget):
                    def handle_right_click(event):
                        if event.button() == Qt.RightButton:
                            # Open color dialog
                            current_color = column_colors[column]
                            qcolor = QColor(current_color)
                            
                            new_color = QColorDialog.getColor(qcolor, columns_menu, f"Choose color for {column}")
                            
                            # If a valid color was selected
                            if new_color.isValid():
                                # Update stored color
                                column_colors[column] = new_color.name()
                                # Update chart color
                                chart.set_color(column, new_color.name())
                                
                                # Update color label
                                color_label_widget.setPixmap(create_color_icon(new_color.name()))
                                
                                # Update chart
                                _update_columns()
                            
                            event.accept()
                    return handle_right_click
                
                # Install mouse press event handler on the color label
                color_label.mousePressEvent = create_color_picker_handler(col, color_label)
                color_label.setCursor(Qt.PointingHandCursor)
        
        # Add a hint to the menu
        columns_menu.addSeparator()
        hint_action = columns_menu.addAction("Right-click on color to change")
        hint_action.setEnabled(False)
        
        # Overlays dropdown button
        overlays_btn = QToolButton()
        overlays_btn.setText("Overlays")
        overlays_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar_layout.addWidget(overlays_btn)
        
        # Create a persistent dropdown menu for overlays
        overlays_menu = PersistentMenu(overlays_btn)
        overlays_btn.setMenu(overlays_menu)
        
        # Dictionary to track overlay visibility
        overlay_checkboxes = {}
        
        # Function to populate overlay menu with custom widget actions
        def _populate_overlays_menu():
            overlays_menu.clear()
            overlay_checkboxes.clear()
            
            # Get all overlays for this dataset
            for ov in DS_MANAGER.overlays(dataset_name):
                # Create widget action
                widget_action = QWidgetAction(overlays_menu)
                
                # Create widget
                action_widget = QWidget()
                action_layout = QHBoxLayout(action_widget)
                action_layout.setContentsMargins(4, 2, 4, 2)
                action_layout.setSpacing(6)
                
                # Add checkbox
                checkbox = QCheckBox()
                checkbox.setChecked(ov.visible)
                action_layout.addWidget(checkbox)
                
                # Add label
                label = QLabel(ov.name)
                action_layout.addWidget(label)
                action_layout.addStretch()
                
                # Set widget as action's widget
                widget_action.setDefaultWidget(action_widget)
                overlays_menu.addAction(widget_action)
                
                # Store references
                overlay_checkboxes[ov.name] = {"action": widget_action, "checkbox": checkbox}
                
                # Connect checkbox changes
                def create_overlay_handler(name, check):
                    def handle_overlay_toggle(state):
                        for ov in DS_MANAGER.overlays(dataset_name):
                            if ov.name == name:
                                ov.visible = (state == Qt.Checked)
                                chart._draw()
                                break
                    return handle_overlay_toggle
                
                checkbox.stateChanged.connect(create_overlay_handler(ov.name, checkbox))
            
            # If no overlays, add a placeholder
            if not overlay_checkboxes:
                placeholder = overlays_menu.addAction("No overlays available")
                placeholder.setEnabled(False)
        
        # Initial population
        _populate_overlays_menu()
        
        # Listen for new overlays
        def _on_new_overlay(ds_name, overlay):
            if ds_name == dataset_name:
                _populate_overlays_menu()
                chart._draw()
        
        DS_MANAGER.add_overlay_listener(_on_new_overlay)
        
        # Add spacer to push navigation tools to the right
        toolbar_spacer = QWidget()
        toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar_layout.addWidget(toolbar_spacer)

        # Add matplotlib navigation toolbar
        self.nav_toolbar = NavigationToolbar(chart, toolbar) # This is the MPL toolbar
        self.nav_toolbar.setIconSize(QSize(16, 16))  # Make icons smaller and more compact
        toolbar_layout.addWidget(self.nav_toolbar)
        
        # Pass the nav_toolbar to the ChartWidget instance AFTER it's created
        chart.mpl_toolbar = self.nav_toolbar
        
        # Add drawing toolbar if available (now that we have nav_toolbar reference)
        self.drawing_toolbar = None
        if HAS_DRAWING_TOOLS and chart.drawing_enabled:
            try:
                self.drawing_toolbar = DrawingToolbar(chart.drawing_manager, self.nav_toolbar, chart)
                main_splitter.addWidget(self.drawing_toolbar)
                
                # Connect toolbar signals
                self.drawing_toolbar.clear_all_requested.connect(self._clear_all_drawings)
                self.drawing_toolbar.save_requested.connect(self._save_drawings)
                self.drawing_toolbar.load_requested.connect(self._load_drawings)
                
                # Monitor matplotlib toolbar for state changes
                if hasattr(self.nav_toolbar, 'pan') and hasattr(self.nav_toolbar, 'zoom'):
                    # Connect to the toolbar actions to detect when they're triggered
                    for action in self.nav_toolbar.actions():
                        if action.text() in ['Pan', 'Zoom']:
                            action.triggered.connect(self._on_mpl_tool_activated)
                
                # Set initial splitter sizes: chart area gets most space, toolbar gets fixed width
                main_splitter.setSizes([800, 220])
                
            except Exception as e:
                print(f"Failed to create drawing toolbar: {e}")
                self.drawing_toolbar = None

        # Add toolbar and splitter to main layout
        layout.addWidget(toolbar)
        layout.addWidget(main_splitter, 1)  # Give stretch priority to chart/table area

        # ------------------------------------------------------------------
        # Signal connections
        # ------------------------------------------------------------------

        def _update_columns():
            # Collect all checked columns
            checked_cols = []
            for col, action in column_checkboxes.items():
                if action["checkbox"].isChecked():
                    checked_cols.append(col)
            
            # Update the chart with selected columns
            chart.set_columns(checked_cols)

        # Connect actions to update function
        for action in column_checkboxes.values():
            action["action"].triggered.connect(_update_columns)

        def _update_mode(text):
            chart.set_chart_mode(text)

        mode_combo.currentTextChanged.connect(_update_mode)

        # Set initial splitter sizes: chart visible, table collapsed
        splitter.setSizes([1, 0])

    def _clear_all_drawings(self):
        """Clear all drawings from the chart."""
        if self.chart.drawing_enabled and self.chart.drawing_manager:
            self.chart.drawing_manager.clear_all_drawings()
            
    def _save_drawings(self):
        """Save drawings to a file."""
        if not self.chart.drawing_enabled or not self.chart.drawing_manager:
            return
            
        try:
            from PySide6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Drawings", f"{self.dataset_name}_drawings.json",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if filename:
                json_data = self.chart.drawing_manager.save_to_json()
                with open(filename, 'w') as f:
                    f.write(json_data)
                    
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Drawings saved to {filename}")
                
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to save drawings:\n{str(e)}")
            
    def _load_drawings(self):
        """Load drawings from a file."""
        if not self.chart.drawing_enabled or not self.chart.drawing_manager:
            return
            
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
            
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Drawings", "",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if filename:
                with open(filename, 'r') as f:
                    json_data = f.read()
                    
                self.chart.drawing_manager.load_from_json(json_data)
                QMessageBox.information(self, "Success", f"Drawings loaded from {filename}")
                
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to load drawings:\n{str(e)}")

    def _on_mpl_tool_activated(self):
        """Called when a matplotlib navigation tool is activated."""
        if self.drawing_toolbar and hasattr(self.nav_toolbar, 'mode'):
            # If a matplotlib tool is active, switch to select mode in drawing toolbar
            if self.nav_toolbar.mode and self.nav_toolbar.mode != '':
                # Find and click the select button
                select_button = self.drawing_toolbar.tool_buttons.get(DrawingToolType.NONE)
                if select_button and not select_button.isChecked():
                    select_button.click()


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
        self._nav_dock = QDockWidget("Datasets (F2)", self)
        self._nav_dock.setObjectName("datasetsDock")
        self._nav_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._nav_list = QListWidget()
        self._nav_list.itemDoubleClicked.connect(self._open_dataset_in_tab)
        
        # Set up the context menu for the dataset list
        self._nav_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._nav_list.customContextMenuRequested.connect(self._show_dataset_context_menu)
        
        self._nav_dock.setWidget(self._nav_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._nav_dock)

        # Listen to dataset additions so navigator stays in sync
        DS_MANAGER.add_listener(self._on_dataset_added)
        # Listen to dataset removals
        DS_MANAGER.add_remove_listener(self._on_dataset_removed)

        # Python console ----------------------------------------------------------
        self._console_dock = None
        self._console_widget = None
        if _HAS_QTCONSOLE:
            try:
                self._console_dock = QDockWidget("Python Console (F4)", self)
                self._console_dock.setObjectName("consoleDock")
                self._console_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)

                self._console_widget = _create_ipython_console(
                    {
                        'ds': DatasetAccessor(DS_MANAGER),
                        'register': register,
                        'register_overlay': register_overlay,
                        'Overlay': Overlay,
                        'update_dataset': update_dataset,
                        'copy_dataset': copy_dataset,
                        'describe_dataset': describe_dataset,
                        'compare_datasets': compare_datasets,
                        'apply_to_dataset': apply_to_dataset,
                        'find_datasets': find_datasets,
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
        self._editor_dock = None
        self._editor_widget = None
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
                    
            self._editor_dock = QDockWidget("Strategy Editor (F3)", self)
            self._editor_dock.setObjectName("editorDock")
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

        # ------------------------------------------------------
        # Strategy dock
        # ------------------------------------------------------

        try:
            from qt_prototype.strategy_ui import StrategyDockWidget  # type: ignore
        except ModuleNotFoundError:  # running as script
            from strategy_ui import StrategyDockWidget  # type: ignore

        strat_widget = StrategyDockWidget(self)
        strat_widget.run_requested.connect(self._run_strategy)

        self._strategy_dock = QDockWidget("Strategies (F5)", self)
        self._strategy_dock.setObjectName("strategiesDock")
        self._strategy_dock.setWidget(strat_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self._strategy_dock)

        # Results dock ---------------------------------------------------
        try:
            from qt_prototype.results_ui import ResultsDockWidget  # type: ignore
        except ModuleNotFoundError:
            from results_ui import ResultsDockWidget  # type: ignore

        self._results_widget = ResultsDockWidget(self)
        self._results_widget.open_requested.connect(self._show_result_tab)
        self._results_dock = QDockWidget("Results (F6)", self)
        self._results_dock.setObjectName("resultsDock")
        self._results_dock.setWidget(self._results_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._results_dock)

        # Menu & actions ----------------------------------------------------------
        open_act = QAction("&Load CSV…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._load_csv)

        toggle_datasets_act = QAction("Toggle &Datasets", self)
        toggle_datasets_act.setShortcut("F2")
        toggle_datasets_act.triggered.connect(lambda: self._toggle_dock(self._nav_dock))

        toggle_console_act = QAction("Toggle &Console", self)
        toggle_console_act.setShortcut("F4")
        toggle_console_act.triggered.connect(self._toggle_console)
        
        toggle_editor_act = QAction("Toggle &Editor", self)
        toggle_editor_act.setShortcut("F3")
        toggle_editor_act.triggered.connect(self._toggle_editor)
        
        toggle_strategies_act = QAction("Toggle &Strategies", self)
        toggle_strategies_act.setShortcut("F5")
        toggle_strategies_act.triggered.connect(lambda: self._toggle_dock(self._strategy_dock))
        
        toggle_results_act = QAction("Toggle &Results", self)
        toggle_results_act.setShortcut("F6")
        toggle_results_act.triggered.connect(lambda: self._toggle_dock(self._results_dock))
        
        new_strategy_act = QAction("&New Strategy", self)
        new_strategy_act.setShortcut("Ctrl+N")
        new_strategy_act.triggered.connect(self._new_strategy)
        
        edit_strategy_act = QAction("&Edit Strategy", self)
        edit_strategy_act.triggered.connect(self._edit_strategy)
        
        # Live chart action
        live_chart_act = QAction("New &Live Chart", self)
        live_chart_act.setShortcut("Ctrl+L")
        live_chart_act.triggered.connect(self._new_live_chart)
        live_chart_act.setEnabled(HAS_LIVE_CHART)
        
        # Layout actions
        reset_layout_act = QAction("&Reset to Default Layout", self)
        reset_layout_act.triggered.connect(self.reset_layout)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        file_menu.addAction(new_strategy_act)
        file_menu.addAction(edit_strategy_act)
        file_menu.addAction(live_chart_act)

        view_menu = self.menuBar().addMenu("&View")
        
        # Add Panels submenu
        panels_menu = view_menu.addMenu("&Panels")
        panels_menu.addAction(toggle_datasets_act)
        panels_menu.addAction(toggle_strategies_act)
        panels_menu.addAction(toggle_results_act)
        
        if self._console_dock:
            panels_menu.addAction(toggle_console_act)
            
        if self._editor_dock:
            panels_menu.addAction(toggle_editor_act)
        
        # Add Layout submenu    
        view_menu.addSeparator()
        view_menu.addAction(reset_layout_act)
        
        # Initial setup - try to restore state or use default
        self._load_state()
        
        # Connect application close events
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self.save_state)
            
    def closeEvent(self, event):
        """Handle window close event to save state."""
        # When the window is closing, save the current state
        self.save_state()
        super().closeEvent(event)
        
    def save_state(self):
        """Save window state including dock positions and visibility."""
        settings = QSettings("BacktestWorkbench", "QtPrototype")
        
        # Save main window state
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
        # Save individual dock visibility states
        settings.setValue("nav_dock_visible", self._nav_dock.isVisible())
        settings.setValue("strategy_dock_visible", self._strategy_dock.isVisible())
        settings.setValue("results_dock_visible", self._results_dock.isVisible())
        
        if self._console_dock:
            settings.setValue("console_dock_visible", self._console_dock.isVisible())
        
        if self._editor_dock:
            settings.setValue("editor_dock_visible", self._editor_dock.isVisible())
            
    def _load_state(self):
        """Load window state and dock visibility from settings."""
        settings = QSettings("BacktestWorkbench", "QtPrototype")
        
        # Check if we have saved settings
        has_state = settings.contains("windowState")
        
        if has_state:
            # First restore the visibility states
            nav_visible = settings.value("nav_dock_visible", True, type=bool)
            strat_visible = settings.value("strategy_dock_visible", True, type=bool)
            results_visible = settings.value("results_dock_visible", True, type=bool)
            
            self._nav_dock.setVisible(nav_visible)
            self._strategy_dock.setVisible(strat_visible)
            self._results_dock.setVisible(results_visible)
            
            if self._console_dock:
                console_visible = settings.value("console_dock_visible", True, type=bool)
                self._console_dock.setVisible(console_visible)
                
            if self._editor_dock:
                editor_visible = settings.value("editor_dock_visible", True, type=bool)
                self._editor_dock.setVisible(editor_visible)
                
            # Then restore geometry and state
            geometry = settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
                
            state = settings.value("windowState")
            if state:
                self.restoreState(state)
        else:
            # Use default layout for first run
            self.reset_layout()

    def restore_state(self):
        """Backward compatibility method for restore_state."""
        self._load_state()
        
    def reset_layout(self):
        """Reset the application layout to default state."""
        # Show all docks
        self._nav_dock.setVisible(True)
        self._strategy_dock.setVisible(True)
        self._results_dock.setVisible(True)
        
        if self._console_dock:
            self._console_dock.setVisible(True)
            
        if self._editor_dock:
            self._editor_dock.setVisible(True)
        
        # Reset dock positions
        self.addDockWidget(Qt.LeftDockWidgetArea, self._nav_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self._strategy_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._results_dock)
        
        if self._console_dock:
            self.addDockWidget(Qt.BottomDockWidgetArea, self._console_dock)
            
            if self._editor_dock:
                self.splitDockWidget(self._console_dock, self._editor_dock, Qt.Horizontal)
        elif self._editor_dock:
            self.addDockWidget(Qt.BottomDockWidgetArea, self._editor_dock)
            
        # Reset window size
        self.resize(1200, 800)

        # Set initial splitter sizes: chart visible, table collapsed
        self.splitter.setSizes([1, 0])

    # ------------------------------------------------------------------
    # Slots / callbacks
    # ------------------------------------------------------------------

    def _load_csv(self):
        import os
        current_dir = os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", current_dir, "CSV files (*.csv)")
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

    def _on_dataset_removed(self, name: str):
        """Callback from DatasetManager when a dataset is removed."""
        # Find and remove the item from the list widget
        for i in range(self._nav_list.count()):
            if self._nav_list.item(i).text() == name:
                self._nav_list.takeItem(i)
                break

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

        self._toggle_dock(self._console_dock)

    def _toggle_editor(self):
        if self._editor_dock is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Editor unavailable", "Failed to initialize code editor. Check console output for details.")
            return
            
        self._toggle_dock(self._editor_dock)

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
        # Create a new tab with the template
        editor = self._editor_widget.editor_tabs.new_file()
        editor.set_text(template)
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
        current_dir = os.getcwd()
        possible_paths = [
            Path(current_dir) / "strategies",
            Path(current_dir) / "btest" / "strategies",
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
            initial_dir = current_dir
        else:
            initial_dir = str(strategy_dir)
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Strategy File", initial_dir,
            "Python Files (*.py);;All Files (*.*)"
        )
        
        if file_path:
            self._editor_widget.editor_tabs.open_file(file_path)
            self._editor_dock.show()
            self._editor_dock.raise_()

    def _new_live_chart(self):
        """Create a new live chart tab."""
        if not HAS_LIVE_CHART:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, 
                "Live Chart Unavailable", 
                "Live chart functionality is not available. Please check that all required dependencies are installed."
            )
            return
            
        try:
            # Create new live chart widget
            live_chart = LiveChartWidget()
            
            # Add as a new tab
            tab_name = live_chart.get_display_name()
            self._tabs.addTab(live_chart, tab_name)
            self._tabs.setCurrentWidget(live_chart)
            
            # Update tab name when currency pair or timeframe changes
            def update_tab_name():
                current_index = self._tabs.indexOf(live_chart)
                if current_index >= 0:
                    self._tabs.setTabText(current_index, live_chart.get_display_name())
            
            # Connect signals to update tab name
            live_chart.pair_combo.currentTextChanged.connect(update_tab_name)
            live_chart.timeframe_combo.currentTextChanged.connect(update_tab_name)
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, 
                "Live Chart Error", 
                f"Failed to create live chart:\n{str(e)}"
            )

    def _show_dataset_context_menu(self, pos):
        item = self._nav_list.itemAt(pos)
        if item:
            name = item.text()
            menu = QMenu()
            
            # Add "Export Dataset" action
            export_act = QAction("Export Dataset", self)
            export_act.triggered.connect(lambda: self._export_dataset(name))
            menu.addAction(export_act)
            
            # Add "Remove Dataset" action
            remove_act = QAction("Remove Dataset", self)
            remove_act.triggered.connect(lambda: self._remove_dataset(name))
            menu.addAction(remove_act)
            
            menu.exec(self._nav_list.mapToGlobal(pos))

    def _export_dataset(self, name):
        """Export the selected dataset to a CSV file."""
        try:
            # Get the dataset (pandas DataFrame)
            df = DS_MANAGER.get(name)
            
            # Get save location from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Dataset",
                f"{name}.csv",  # Default filename
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                try:
                    # Export to CSV
                    df.to_csv(file_path, index=True)
                    self.statusBar().showMessage(f"Dataset exported to {file_path}", 3000)
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to export dataset: {str(e)}")
        except KeyError:
            QMessageBox.warning(self, "Export Error", f"Dataset '{name}' not found.")

    def _remove_dataset(self, name):
        """Remove a dataset from the dataset manager."""
        from PySide6.QtWidgets import QMessageBox
        
        # Ask for confirmation before removing
        reply = QMessageBox.question(
            self, 
            "Remove Dataset",
            f"Are you sure you want to remove dataset '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Close any open tabs showing this dataset
            for i in range(self._tabs.count()):
                widget = self._tabs.widget(i)
                if isinstance(widget, Workspace) and widget.dataset_name == name:
                    self._tabs.removeTab(i)
                    break
            
            # Remove the dataset
            if DS_MANAGER.remove_dataset(name):
                # No need to update the list view manually as we've added a listener
                pass

    # ------------------------------------------------------------------
    # Dock visibility helpers
    # ------------------------------------------------------------------
    
    def _toggle_dock(self, dock):
        """Toggle visibility of a dock widget."""
        if dock:
            dock.setVisible(not dock.isVisible())


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
            "Available objects and functions:\n\n"
            "Dataset Access:\n"
            "  ds                       → Dataset accessor with multiple access methods:\n"
            "    ds[0], ds[1], ...      → Access datasets by numerical index\n"
            "    ds.a, ds.b, ...        → Access datasets by letter alias (first 26 datasets)\n"
            "    ds['dataset_name']     → Access datasets by name\n"
            "    ds.list()              → List all datasets with indices and aliases\n"
            "    ds.names()             → Return a list of all dataset names\n"
            "    ds.get_name(index)     → Get dataset name from its index\n"
            "    ds.get_index(name)     → Get dataset index from its name\n\n"
            "Dataset Management:\n"
            "  register(df, name=None)  → Add a new DataFrame to the dataset manager\n"
            "                             Example: register(my_df, 'my_dataset')\n\n"
            "  update_dataset(name, df) → Update an existing dataset with new data\n"
            "                             Example: update_dataset('eurusd', updated_df)\n\n"
            "  copy_dataset(name, new_name=None) → Create a copy of a dataset\n"
            "                             Example: copy_dataset('eurusd', 'eurusd_modified')\n"
            "                             If new_name is omitted, creates name automatically\n\n"
            "Dataset Analysis:\n"
            "  describe_dataset(name_or_index) → Print summary statistics for a dataset\n"
            "                             Example: describe_dataset('eurusd') or describe_dataset(0)\n\n"
            "  compare_datasets(name1, name2) → Compare two datasets and highlight differences\n"
            "                             Example: compare_datasets('eurusd', 'eurusd_modified')\n"
            "                             Works with indices too: compare_datasets(0, 1)\n\n"
            "Data Transformation:\n"
            "  apply_to_dataset(name, func, columns=None, new_columns=None, result_name=None)\n"
            "                             → Apply a function to dataset columns\n"
            "                             Example: apply_to_dataset('eurusd', lambda x: x.rolling(20).mean(),\n"
            "                                      columns=['Close'], new_columns=['MA20'])\n\n"
            "  find_datasets(criteria=None, column_pattern=None, name_pattern=None)\n"
            "                             → Find datasets matching specific criteria\n"
            "                             Example: find_datasets(criteria=lambda df: len(df) > 1000)\n"
            "                             Example: find_datasets(column_pattern='close|open')\n\n"
            "Visualization:\n"
            "  register_overlay(dataset, name, df, style=None) → Add an overlay layer to a chart\n"
            "                             Example: register_overlay('eurusd', 'Support', support_df)\n"
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
                "Available objects and functions:\n\n"
                "Dataset Access:\n"
                "  ds                       → Dataset accessor with multiple access methods:\n"
                "    ds[0], ds[1], ...      → Access datasets by numerical index\n"
                "    ds.a, ds.b, ...        → Access datasets by letter alias (first 26 datasets)\n"
                "    ds['dataset_name']     → Access datasets by name\n"
                "    ds.list()              → List all datasets with indices and aliases\n"
                "    ds.names()             → Return a list of all dataset names\n"
                "    ds.get_name(index)     → Get dataset name from its index\n"
                "    ds.get_index(name)     → Get dataset index from its name\n\n"
                "Dataset Management:\n"
                "  register(df, name=None)  → Add a new DataFrame to the dataset manager\n"
                "                             Example: register(my_df, 'my_dataset')\n\n"
                "  update_dataset(name, df) → Update an existing dataset with new data\n"
                "                             Example: update_dataset('eurusd', updated_df)\n\n"
                "  copy_dataset(name, new_name=None) → Create a copy of a dataset\n"
                "                             Example: copy_dataset('eurusd', 'eurusd_modified')\n"
                "                             If new_name is omitted, creates name automatically\n\n"
                "Dataset Analysis:\n"
                "  describe_dataset(name_or_index) → Print summary statistics for a dataset\n"
                "                             Example: describe_dataset('eurusd') or describe_dataset(0)\n\n"
                "  compare_datasets(name1, name2) → Compare two datasets and highlight differences\n"
                "                             Example: compare_datasets('eurusd', 'eurusd_modified')\n"
                "                             Works with indices too: compare_datasets(0, 1)\n\n"
                "Data Transformation:\n"
                "  apply_to_dataset(name, func, columns=None, new_columns=None, result_name=None)\n"
                "                             → Apply a function to dataset columns\n"
                "                             Example: apply_to_dataset('eurusd', lambda x: x.rolling(20).mean(),\n"
                "                                      columns=['Close'], new_columns=['MA20'])\n\n"
                "  find_datasets(criteria=None, column_pattern=None, name_pattern=None)\n"
                "                             → Find datasets matching specific criteria\n"
                "                             Example: find_datasets(criteria=lambda df: len(df) > 1000)\n"
                "                             Example: find_datasets(column_pattern='close|open')\n\n"
                "Visualization:\n"
                "  register_overlay(dataset, name, df, style=None) → Add an overlay layer to a chart\n"
                "                             Example: register_overlay('eurusd', 'Support', support_df)\n"
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
