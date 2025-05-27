from __future__ import annotations

"""Enhanced live chart widget with streaming and polling support.

This module provides an enhanced live chart widget that supports both
polling and streaming modes for real-time market data visualization.
"""

import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QSize
from PySide6.QtGui import QAction, QColor, QBrush, QPalette, QPixmap, QPainter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QFrame, QSizePolicy, QToolButton, QMenu, QCheckBox, QSpinBox,
    QMessageBox, QProgressBar, QGroupBox, QFormLayout, QLineEdit,
    QColorDialog, QWidgetAction, QScrollArea, QSplitter, QButtonGroup,
    QRadioButton
)

# from qt_prototype.main import NavigationToolbar, ChartWidget # <-- REMOVE TOP-LEVEL IMPORT
from qt_prototype.drawing_toolbar import DrawingToolbar
from qt_prototype.drawing_tools import HAS_DRAWING_TOOLS, DrawingToolType # Import DrawingToolType for _set_drawing_tool

from qt_prototype.oanda_api import get_oanda_api, set_api_key, OandaAPI
from qt_prototype.oanda_streaming import StreamingPriceManager


class EnhancedLiveChartWidget(QWidget):
    """Enhanced live chart widget with streaming support."""
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        # Current state
        self.current_instrument = "EUR_USD"
        self.current_timeframe = "M15"
        self.current_df = None
        self.chart = None
        self.nav_toolbar = None
        self.drawing_toolbar = None
        self.dataset_name = None
        
        # Streaming mode only
        self.data_mode = "streaming"
        
        # Streaming components
        self.streaming_manager = StreamingPriceManager()
        
        # Price tracking (for streaming)
        self.current_price = None
        self.price_change = 0.0
        self.price_change_percent = 0.0
        
        self._setup_ui()
        self._connect_signals()
        
        # Start streaming mode after UI is set up
        self._start_streaming()
        
    def _start_streaming(self):
        """Initialize streaming mode and fetch initial historical data."""
        # This method is called after _setup_ui, so UI elements are available.
        self.connection_status_label.setText("Connecting...")
        self.connection_status_label.setStyleSheet("color: orange; font-size: 10px;")
        self.progress_bar.setVisible(True) # Show progress for historical fetch

        # Fetch historical data first, then start streaming connection
        self._fetch_historical_for_streaming() 
        # _fetch_historical_for_streaming will call _update_chart, which enables add_dataset_btn
        # and hides the progress_bar if successful.
        
        # Attempt to start the actual Oanda price stream
        if self.streaming_manager.start_streaming([self.current_instrument]):
            self.streaming_manager.subscribe_to_candles(self.current_instrument, self.current_timeframe)
            # Connection status will be updated by the _on_streaming_connection_status signal handler
        else:
            # If start_stream itself fails (e.g., API key issue before even trying to connect)
            self.connection_status_label.setText("Stream Init Failed")
            self.connection_status_label.setStyleSheet("color: red; font-size: 10px;")
            self.progress_bar.setVisible(False)
        
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create toolbar (should be the cleaned-up version)
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # Create and add the price bar for live tick display
        price_bar = self._create_price_bar()
        layout.addWidget(price_bar) 
        
        # Create main horizontal splitter for chart and drawing toolbar
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create chart placeholder
        self.chart_container = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add loading indicator
        self.loading_label = QLabel("Loading chart data...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 14px; color: gray;")
        self.chart_layout.addWidget(self.loading_label)
        
        # Add chart container to main splitter
        main_splitter.addWidget(self.chart_container)
        
        # Initialize drawing toolbar as None (will be created when chart is ready)
        self.drawing_toolbar = None
        
        # Add main splitter to layout
        layout.addWidget(main_splitter, 1)
        
        # Store reference to main splitter for later use
        self.main_splitter = main_splitter
        
    def _create_toolbar(self) -> QFrame:
        """Create the toolbar with controls for streaming mode."""
        toolbar = QFrame()
        toolbar.setFrameShape(QFrame.StyledPanel)
        toolbar.setMaximumHeight(40)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(6, 4, 6, 4)
        toolbar_layout.setSpacing(10)
        
        # Currency pair selector
        toolbar_layout.addWidget(QLabel("Pair:"))
        self.pair_combo = QComboBox()
        self.pair_combo.addItems(OandaAPI.CURRENCY_PAIRS)
        self.pair_combo.setCurrentText(self.current_instrument)
        self.pair_combo.setMinimumWidth(100)
        toolbar_layout.addWidget(self.pair_combo)
        
        # Timeframe selector
        toolbar_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        for tf, info in OandaAPI.TIMEFRAMES.items():
            self.timeframe_combo.addItem(info['label'], tf)
        self.timeframe_combo.setCurrentText(OandaAPI.TIMEFRAMES[self.current_timeframe]['label'])
        self.timeframe_combo.setMinimumWidth(100)
        toolbar_layout.addWidget(self.timeframe_combo)
        
        # Candle count selector (for historical data)
        toolbar_layout.addWidget(QLabel("History:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(50, 5000)
        self.count_spin.setValue(500) # Default historical candles
        self.count_spin.setSuffix(" bars")
        self.count_spin.setMinimumWidth(80)
        self.count_spin.setToolTip("Number of historical candles to load for context")
        toolbar_layout.addWidget(self.count_spin)
        
        # Refresh button (for historical data segment)
        self.refresh_btn = QPushButton("Refresh History")
        self.refresh_btn.setMaximumWidth(120)
        self.refresh_btn.setToolTip("Manually re-fetch the historical data segment")
        toolbar_layout.addWidget(self.refresh_btn)
        
        # Connection status indicator (for streaming)
        self.connection_status_label = QLabel("Idle") # Initial state before _start_streaming
        self.connection_status_label.setStyleSheet("color: gray; font-size: 10px;")
        toolbar_layout.addWidget(self.connection_status_label)
        
        toolbar_layout.addStretch()
        
        # Add to dataset button
        self.add_dataset_btn = QPushButton("Add to Datasets")
        self.add_dataset_btn.setEnabled(False) # Enabled once chart data is loaded
        toolbar_layout.addWidget(self.add_dataset_btn)
        
        # API settings button
        self.api_settings_btn = QPushButton("API Settings")
        toolbar_layout.addWidget(self.api_settings_btn)
        
        # Clear cache button
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.setMaximumWidth(100)
        self.clear_cache_btn.setToolTip("Clear cached historical data for current instrument/timeframe")
        toolbar_layout.addWidget(self.clear_cache_btn)
        
        # Progress bar (for historical fetch)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(4)
        toolbar_layout.addWidget(self.progress_bar)
        
        return toolbar
        
    def _create_price_bar(self) -> QFrame:
        """Create the real-time price display bar."""
        price_bar = QFrame()
        price_bar.setFrameShape(QFrame.StyledPanel)
        price_bar.setMaximumHeight(30)
        price_bar.setStyleSheet("background-color: #f0f0f0;")
        
        layout = QHBoxLayout(price_bar)
        layout.setContentsMargins(10, 4, 10, 4)
        
        # Current price display
        self.price_label = QLabel("--")
        self.price_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.price_label)
        
        # Price change display
        self.change_label = QLabel("--")
        self.change_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.change_label)
        
        # Bid/Ask spread (for streaming mode)
        self.spread_label = QLabel("--")
        self.spread_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self.spread_label)
        
        layout.addStretch()
        
        # Last update time
        self.last_update_label = QLabel("--")
        self.last_update_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self.last_update_label)
        
        return price_bar
        
    def _connect_signals(self):
        """Connect UI signals."""
        # Instrument and timeframe changes
        self.pair_combo.currentTextChanged.connect(self._on_pair_changed)
        self.timeframe_combo.currentTextChanged.connect(self._on_timeframe_changed)
        self.count_spin.valueChanged.connect(self._on_count_changed)
        
        # Refresh controls
        self.refresh_btn.clicked.connect(self._refresh_historical_data)
        
        # Other buttons
        self.add_dataset_btn.clicked.connect(self._add_to_datasets)
        self.api_settings_btn.clicked.connect(self._show_api_settings)
        self.clear_cache_btn.clicked.connect(self._clear_cache)
        
        # Streaming signals
        self.streaming_manager.candle_update.connect(self._on_streaming_candle_update)
        self.streaming_manager.tick_update.connect(self._on_streaming_tick_update)
        self.streaming_manager.streaming_api.connection_status.connect(self._on_streaming_connection_status)
        self.streaming_manager.streaming_api.error_occurred.connect(self._on_streaming_error)
        
    def _refresh_historical_data(self):
        """Refresh historical data manually."""
        self._fetch_historical_for_streaming()
        
    def _on_count_changed(self, count: int):
        """Handle candle count change."""
        # Refresh historical data with new count
        self._fetch_historical_for_streaming()
        
    def _on_pair_changed(self, pair: str):
        """Handle currency pair change."""
        if pair != self.current_instrument:
            self.current_instrument = pair
            self._reset_price_tracking()
            self._update_dataset_name()
            
            # Update streaming subscription
            self.streaming_manager.unsubscribe_from_candles(self.current_instrument, self.current_timeframe)
            self.streaming_manager.stop_streaming()
            self._fetch_historical_for_streaming()
            self.streaming_manager.start_streaming([self.current_instrument])
            self.streaming_manager.subscribe_to_candles(self.current_instrument, self.current_timeframe)
                
    def _on_timeframe_changed(self, timeframe_label: str):
        """Handle timeframe change."""
        # Find timeframe code from label
        for tf, info in OandaAPI.TIMEFRAMES.items():
            if info['label'] == timeframe_label:
                if tf != self.current_timeframe:
                    old_timeframe = self.current_timeframe
                    self.current_timeframe = tf
                    self._update_dataset_name()
                    
                    # Update streaming subscription
                    self.streaming_manager.unsubscribe_from_candles(self.current_instrument, old_timeframe)
                    self._fetch_historical_for_streaming()
                    self.streaming_manager.subscribe_to_candles(self.current_instrument, self.current_timeframe)
                break
                
    def _update_dataset_name(self):
        """Update the dataset name when instrument or timeframe changes."""
        self.dataset_name = f"{self.current_instrument}_{self.current_timeframe}_live"
        
    def _reset_price_tracking(self):
        """Reset price tracking variables."""
        self.current_price = None
        self.price_change = 0.0
        self.price_change_percent = 0.0
        self.price_label.setText("--")
        self.change_label.setText("--")
        self.spread_label.setText("--")
        self.last_update_label.setText("--")
        
    def _fetch_historical_for_streaming(self):
        """Fetch historical data before starting streaming."""
        self.progress_bar.setVisible(True)
        self.refresh_btn.setEnabled(False)
        try:
            from qt_prototype.oanda_api import OandaAPI
            
            api = OandaAPI()
            count = self.count_spin.value()
            
            df = api.get_candles(
                instrument=self.current_instrument,
                timeframe=self.current_timeframe,
                count=count,
                force_refresh=True # Always force refresh for manual call
            )
            
            if df is not None and not df.empty:
                df.columns = df.columns.str.lower()
                self.current_df = df
                self._update_chart() # This will create/update the chart
                self.add_dataset_btn.setEnabled(True)
            else:
                # Handle no data received for historical fetch
                if self.chart: # Clear existing chart if any
                    self.chart_layout.removeWidget(self.chart)
                    self.chart.deleteLater()
                    self.chart = None
                    if self.nav_toolbar:
                        self.chart_layout.removeWidget(self.nav_toolbar)
                        self.nav_toolbar.deleteLater()
                        self.nav_toolbar = None
                    if self.drawing_toolbar:
                        self.main_splitter.widget(1).deleteLater() # Remove drawing toolbar from splitter
                        self.drawing_toolbar = None
                self.loading_label.setText(f"No historical data for {self.current_instrument} {self.current_timeframe}")
                self.loading_label.setVisible(True)
                self.add_dataset_btn.setEnabled(False)

        except Exception as e:
            print(f"Failed to fetch historical data: {e}")
            self.connection_status_label.setText("Hist. Data Failed")
            self.connection_status_label.setStyleSheet("color: red; font-size: 10px;")
            # Optionally show error in loading_label or a QMessageBox
            self.loading_label.setText(f"Error loading history: {e}")
            self.loading_label.setVisible(True)
        finally:
            self.progress_bar.setVisible(False)
            self.refresh_btn.setEnabled(True)
        
    def _on_streaming_candle_update(self, instrument: str, timeframe: str, candle_data: Dict):
        """Handle streaming candle updates."""
        if instrument == self.current_instrument and timeframe == self.current_timeframe:
            # Convert candle data to DataFrame format
            timestamp = candle_data['timestamp']
            
            # Create single-row DataFrame
            df_data = {
                'open': [candle_data['open']],
                'high': [candle_data['high']],
                'low': [candle_data['low']],
                'close': [candle_data['close']],
                'volume': [candle_data['volume']]
            }
            
            new_candle_df = pd.DataFrame(df_data, index=[timestamp])
            
            # Update current DataFrame
            if self.current_df is not None and not self.current_df.empty:
                # Check if this is an update to the last candle or a new candle
                last_timestamp = self.current_df.index[-1]
                
                if timestamp == last_timestamp:
                    # Update existing candle
                    self.current_df.iloc[-1] = new_candle_df.iloc[0]
                else:
                    # Add new candle
                    self.current_df = pd.concat([self.current_df, new_candle_df])
                    
                    # Keep only recent candles (use count from spinner)
                    max_candles = self.count_spin.value()
                    if len(self.current_df) > max_candles:
                        self.current_df = self.current_df.tail(max_candles)
            else:
                self.current_df = new_candle_df
                
            # Update chart efficiently
            self._update_chart_data_only()
            
    def _on_streaming_tick_update(self, instrument: str, tick_data: Dict):
        """Handle streaming tick updates."""
        if instrument == self.current_instrument:
            # Update real-time price display
            bids = tick_data.get('bids', [])
            asks = tick_data.get('asks', [])
            
            if bids and asks:
                bid_price = float(bids[0]['price'])
                ask_price = float(asks[0]['price'])
                mid_price = (bid_price + ask_price) / 2
                spread = ask_price - bid_price
                
                # Calculate price change
                if self.current_price is not None:
                    self.price_change = mid_price - self.current_price
                    if self.current_price != 0:
                        self.price_change_percent = (self.price_change / self.current_price) * 100
                        
                self.current_price = mid_price
                
                # Update price display
                self._update_price_display(mid_price, bid_price, ask_price, spread)
                
                # Update last update time
                timestamp = datetime.fromisoformat(tick_data['time'].replace('Z', '+00:00'))
                self.last_update_label.setText(f"Last: {timestamp.strftime('%H:%M:%S')}")
                
    def _update_price_display(self, mid_price: float, bid_price: float, ask_price: float, spread: float):
        """Update the real-time price display."""
        # Format price based on instrument
        if 'JPY' in self.current_instrument:
            price_format = "{:.3f}"
            spread_format = "{:.1f}"
        else:
            price_format = "{:.5f}"
            spread_format = "{:.1f}"
            
        # Update price label
        self.price_label.setText(price_format.format(mid_price))
        
        # Update change label with color coding
        if self.price_change > 0:
            change_text = f"+{price_format.format(self.price_change)} (+{self.price_change_percent:.2f}%)"
            self.change_label.setText(change_text)
            self.change_label.setStyleSheet("font-size: 12px; color: green;")
            self.price_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        elif self.price_change < 0:
            change_text = f"{price_format.format(self.price_change)} ({self.price_change_percent:.2f}%)"
            self.change_label.setText(change_text)
            self.change_label.setStyleSheet("font-size: 12px; color: red;")
            self.price_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        else:
            self.change_label.setText("--")
            self.change_label.setStyleSheet("font-size: 12px; color: gray;")
            self.price_label.setStyleSheet("font-size: 16px; font-weight: bold; color: black;")
            
        # Update spread display
        spread_pips = spread * (10000 if 'JPY' not in self.current_instrument else 100)
        self.spread_label.setText(f"Spread: {spread_format.format(spread_pips)} pips")
        
    def _update_chart(self):
        """Update the chart with current data."""
        # Local import to break circular dependency
        from qt_prototype.main import ChartWidget, NavigationToolbar

        if self.current_df is None or self.current_df.empty:
            self.loading_label.setText(f"No data available for {self.current_instrument} {self.current_timeframe}")
            self.loading_label.setVisible(True)
            if self.chart: 
                if self.nav_toolbar and self.chart_layout.indexOf(self.nav_toolbar) != -1:
                    self.chart_layout.removeWidget(self.nav_toolbar)
                    self.nav_toolbar.deleteLater()
                    self.nav_toolbar = None
                if self.chart and self.chart_layout.indexOf(self.chart) != -1:
                    self.chart_layout.removeWidget(self.chart)
                    self.chart.deleteLater()
                    self.chart = None
                if self.drawing_toolbar and self.main_splitter.indexOf(self.drawing_toolbar) != -1: 
                    self.drawing_toolbar.deleteLater() 
                    self.drawing_toolbar = None
            self.add_dataset_btn.setEnabled(False)
            return
            
        recreating_chart = self.chart is None

        if recreating_chart:
            self._update_dataset_name() 
            
            self.chart = ChartWidget(
                df=self.current_df, 
                dataset_name=self.dataset_name,
                parent=self.chart_container 
            )
            
            self.loading_label.setVisible(False)
            self.chart_layout.addWidget(self.chart)
            
            self.nav_toolbar = self.chart.mpl_toolbar 
            if self.nav_toolbar:
                if self.nav_toolbar.parent() != self.chart_container:
                    self.nav_toolbar.setParent(self.chart_container)
            else:
                self.nav_toolbar = NavigationToolbar(self.chart, self.chart_container)
                self.chart_layout.addWidget(self.nav_toolbar) 
                self.chart.mpl_toolbar = self.nav_toolbar

            if HAS_DRAWING_TOOLS and self.chart.drawing_enabled and self.chart.drawing_manager:
                try:
                    if self.drawing_toolbar and self.main_splitter.indexOf(self.drawing_toolbar) != -1:
                        self.drawing_toolbar.deleteLater()
                        self.drawing_toolbar = None

                    self.drawing_toolbar = DrawingToolbar(
                        drawing_manager=self.chart.drawing_manager,    
                        matplotlib_toolbar=self.nav_toolbar,           
                        chart_widget=self.chart                        
                    )
                    self.main_splitter.addWidget(self.drawing_toolbar)
                    
                    self.drawing_toolbar.clear_all_requested.connect(self._clear_all_drawings_original)
                    self.drawing_toolbar.save_requested.connect(self._save_drawings_original)
                    self.drawing_toolbar.load_requested.connect(self._load_drawings_original)
                    
                    if self.main_splitter.count() >= 2:
                        splitter_width = self.main_splitter.width()
                        if splitter_width > 50: 
                            chart_width = int(splitter_width * 0.8)
                            toolbar_widget_width = splitter_width - chart_width
                            self.main_splitter.setSizes([chart_width, toolbar_widget_width])
                        else:
                            self.main_splitter.setSizes([600,150]) 
                except Exception as e:
                    print(f"Failed to create original drawing toolbar for live chart: {e}")
                    import traceback
                    traceback.print_exc()
                    self.drawing_toolbar = None
            
            if all(col in self.current_df.columns for col in ['open', 'high', 'low', 'close']):
                self.chart.set_chart_mode("Candlestick")
            else:
                self.chart.set_chart_mode("Line")
                numeric_cols = [c for c in self.current_df.columns if pd.api.types.is_numeric_dtype(self.current_df[c])]
                if numeric_cols:
                    self.chart.set_columns([numeric_cols[0]])
            
            self.add_dataset_btn.setEnabled(True)
            
        else: 
            self._update_dataset_name()
            if self.chart: 
                self.chart.dataset_name = self.dataset_name 
                self.chart._df = self.current_df
                self.chart.update_drawing_data(self.current_df) 
                self.chart._draw()

        if self.chart:
            self.chart.setFocusPolicy(Qt.StrongFocus)

    def _update_chart_data_only(self):
        """Efficiently update chart data without full redraw (for streaming)."""
        if self.chart is not None and self.current_df is not None:
            self.chart._df = self.current_df
            self.chart.update_drawing_data(self.current_df)
            
            if not (hasattr(self.chart, '_drawing_in_progress') and self.chart._drawing_in_progress):
                if not (self.nav_toolbar and hasattr(self.nav_toolbar, 'mode') and self.nav_toolbar.mode): # Check nav_toolbar exists and has mode
                    self.chart._draw()
    
    def _clear_all_drawings_original(self):
        """Clear all drawings using DrawingManager via original toolbar's request."""
        if self.chart and self.chart.drawing_manager:
            self.chart.drawing_manager.clear_all_drawings()
            
    def _save_drawings_original(self):
        """Save current drawings using DrawingManager via original toolbar's request."""
        if self.chart and self.chart.drawing_manager:
            from PySide6.QtWidgets import QFileDialog # Keep import local to method
            # Suggest a filename based on the current chart
            suggested_filename = f"{self.dataset_name}_drawings.json" if self.dataset_name else "drawings.json"
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Drawings", suggested_filename,
                "JSON Files (*.json);;All Files (*)"
            )
            if filename:
                json_data = self.chart.drawing_manager.save_to_json()
                with open(filename, 'w') as f:
                    f.write(json_data)
                QMessageBox.information(self, "Success", f"Drawings saved to {filename}")

    def _load_drawings_original(self):
        """Load saved drawings using DrawingManager via original toolbar's request."""
        if self.chart and self.chart.drawing_manager:
            from PySide6.QtWidgets import QFileDialog # Keep import local to method
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Drawings", "",
                "JSON Files (*.json);;All Files (*)"
            )
            if filename:
                try:
                    with open(filename, 'r') as f:
                        json_data = f.read()
                    self.chart.drawing_manager.load_from_json(json_data)
                    QMessageBox.information(self, "Success", f"Drawings loaded from {filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Failed to load drawings: {e}")

    def _add_to_datasets(self):
        """Add current data to datasets."""
        if self.current_df is None or self.current_df.empty:
            return
            
        from qt_prototype.main import DS_MANAGER
        
        # Create a unique dataset name
        dataset_name = f"{self.current_instrument}_{self.current_timeframe}_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add to dataset manager
        DS_MANAGER.add_dataframe(self.current_df.copy(), dataset_name)
        
        # Show confirmation
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, 
            "Dataset Added", 
            f"Live streaming data saved as dataset '{dataset_name}' with {len(self.current_df)} candles."
        )
        
    def _show_api_settings(self):
        """Show API settings dialog."""
        try:
            from qt_prototype.live_chart import ApiSettingsDialog
            dialog = ApiSettingsDialog(self)
            dialog.exec()
        except ImportError:
            QMessageBox.information(self, "API Settings", "API settings dialog not available.")
        
    def _clear_cache(self):
        """Clear cached data."""
        try:
            api = get_oanda_api()
            cache_key = f"{self.current_instrument}_{self.current_timeframe}"
            if hasattr(api, '_cache') and cache_key in api._cache:
                del api._cache[cache_key]
                QMessageBox.information(self, "Cache Cleared", f"Cleared cache for {self.current_instrument} {self.current_timeframe}")
            else:
                QMessageBox.information(self, "Cache", "No cached data found for current instrument/timeframe")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear cache: {e}")
        
    def get_display_name(self) -> str:
        """Get display name for the tab."""
        timeframe_label = OandaAPI.TIMEFRAMES.get(self.current_timeframe, {}).get('label', self.current_timeframe)
        return f"🔴 {self.current_instrument} {timeframe_label}"
        
    def closeEvent(self, event):
        """Handle widget close event."""
        # Stop streaming if active
        self.streaming_manager.stop_streaming()
        
        super().closeEvent(event)
        
    def _on_streaming_connection_status(self, status: str):
        """Handle streaming connection status changes."""
        self.progress_bar.setVisible(False) # Hide progress bar once connection attempt is resolved
        if status == "connecting":
            self.connection_status_label.setText("Connecting...")
            self.connection_status_label.setStyleSheet("color: orange; font-size: 10px;")
        elif status == "connected":
            self.connection_status_label.setText("Streaming Live")
            self.connection_status_label.setStyleSheet("color: green; font-size: 10px; font-weight: bold;")
        elif status == "disconnected":
            self.connection_status_label.setText("Disconnected")
            self.connection_status_label.setStyleSheet("color: red; font-size: 10px;")
        elif status.startswith("reconnecting"):
            self.connection_status_label.setText(f"Reconnecting...")
            self.connection_status_label.setStyleSheet("color: orange; font-size: 10px;")
        else:
            self.connection_status_label.setText(status) # Catchall for other statuses
            self.connection_status_label.setStyleSheet("color: gray; font-size: 10px;")
            
    def _on_streaming_error(self, error_msg: str):
        """Handle streaming errors."""
        print(f"Streaming error in live chart: {error_msg}")
        self.connection_status_label.setText("Stream Error")
        self.connection_status_label.setStyleSheet("color: red; font-size: 10px; font-weight: bold;")
        self.progress_bar.setVisible(False)
        # Optionally show a QMessageBox for critical errors
        # QMessageBox.warning(self, "Streaming Error", error_msg) 