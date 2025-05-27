"""Oanda Streaming API integration for real-time market data.

This module provides streaming functionality to receive real-time price updates
from Oanda's streaming API, complementing the existing REST API implementation.
"""

import os
import json
import requests
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Callable, List, Set
from pathlib import Path
import queue
import logging

from PySide6.QtCore import QObject, Signal, QThread, QTimer

# Import existing API for configuration
from qt_prototype.oanda_api import OandaAPI


class OandaStreamingAPI(QObject):
    """Oanda streaming API client for real-time price updates."""
    
    # Signals for Qt integration
    price_update = Signal(dict)  # Emits price data
    heartbeat = Signal(dict)     # Emits heartbeat data
    connection_status = Signal(str)  # Emits connection status
    error_occurred = Signal(str)     # Emits error messages
    
    def __init__(self, api_key: str = None, environment: str = None):
        """Initialize Oanda streaming API client.
        
        Parameters
        ----------
        api_key : str, optional
            Oanda API key. If None, will try to read from environment variable OANDA_API_KEY
        environment : str, optional
            'practice' for demo account, 'live' for real account.
            If None, will try to read from OANDA_ENVIRONMENT (defaults to 'practice')
        """
        super().__init__()
        
        self.api_key = api_key or os.getenv('OANDA_API_KEY')
        self.environment = environment or os.getenv('OANDA_ENVIRONMENT', 'practice')
        
        if self.environment == 'practice':
            self.stream_url = 'https://stream-fxpractice.oanda.com'
        else:
            self.stream_url = 'https://stream-fxtrade.oanda.com'
            
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/octet-stream'
        }
        
        # Get timeout from environment or use default
        self.timeout = int(os.getenv('OANDA_TIMEOUT', '30'))
        
        # Streaming state
        self._streaming = False
        self._stream_thread = None
        self._response = None
        self._subscribed_instruments = set()
        
        # Connection monitoring
        self._last_heartbeat = None
        self._heartbeat_timeout = 10  # seconds
        self._reconnect_delay = 5     # seconds
        self._max_reconnect_attempts = 5
        self._reconnect_attempts = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def start_stream(self, instruments: List[str], include_heartbeats: bool = True) -> bool:
        """Start streaming prices for specified instruments.
        
        Parameters
        ----------
        instruments : List[str]
            List of instrument names (e.g., ['EUR_USD', 'GBP_USD'])
        include_heartbeats : bool, optional
            Whether to include heartbeat messages (default: True)
            
        Returns
        -------
        bool
            True if stream started successfully, False otherwise
        """
        if self._streaming:
            self.logger.warning("Stream already running")
            return False
            
        if not self.api_key:
            self.error_occurred.emit("API key is required")
            return False
            
        if not instruments:
            self.error_occurred.emit("At least one instrument must be specified")
            return False
            
        self._subscribed_instruments = set(instruments)
        
        # Start streaming thread
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_worker,
            args=(instruments, include_heartbeats),
            daemon=True
        )
        self._stream_thread.start()
        
        self.connection_status.emit("connecting")
        return True
        
    def stop_stream(self):
        """Stop the streaming connection."""
        if not self._streaming:
            return
            
        self._streaming = False
        
        # Close the response stream if it exists
        if self._response:
            try:
                self._response.close()
            except:
                pass
                
        # Wait for thread to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)
            
        self.connection_status.emit("disconnected")
        
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._streaming
        
    def get_subscribed_instruments(self) -> Set[str]:
        """Get currently subscribed instruments."""
        return self._subscribed_instruments.copy()
        
    def _stream_worker(self, instruments: List[str], include_heartbeats: bool):
        """Worker thread for handling the streaming connection."""
        while self._streaming:
            try:
                self._connect_and_stream(instruments, include_heartbeats)
            except Exception as e:
                self.logger.error(f"Stream error: {e}")
                self.error_occurred.emit(f"Stream error: {e}")
                
                # Clean up response object
                if self._response:
                    try:
                        self._response.close()
                    except:
                        pass
                    self._response = None
                
                if self._streaming:
                    self.connection_status.emit("disconnected")
                    self._handle_reconnection()
                    
    def _connect_and_stream(self, instruments: List[str], include_heartbeats: bool):
        """Establish connection and process streaming data."""
        # Prepare request parameters
        params = {
            'instruments': ','.join(instruments),
            'snapshot': 'true'  # Include initial snapshot
        }
        
        if include_heartbeats:
            params['includeHeartbeats'] = 'true'
            
        # Get account ID (required for streaming endpoint)
        account_id = self._get_account_id()
        if not account_id:
            self.error_occurred.emit("Could not retrieve account ID")
            return
            
        url = f"{self.stream_url}/v3/accounts/{account_id}/pricing/stream"
        
        self.logger.info(f"Connecting to stream: {url}")
        self.logger.info(f"Instruments: {instruments}")
        
        # Clean up any existing response
        if self._response:
            try:
                self._response.close()
            except:
                pass
            self._response = None
        
        # Make streaming request
        try:
            self._response = requests.get(
                url,
                headers=self.headers,
                params=params,
                stream=True,
                timeout=self.timeout
            )
            
            self._response.raise_for_status()
            self.connection_status.emit("connected")
            self._reconnect_attempts = 0
            
            # Process streaming data
            self._process_stream()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to establish streaming connection: {e}")
            if self._response:
                try:
                    self._response.close()
                except:
                    pass
                self._response = None
            raise
        
    def _process_stream(self):
        """Process incoming streaming data."""
        if not self._response:
            self.logger.error("No response object available for streaming")
            return
            
        try:
            for line in self._response.iter_lines(decode_unicode=True):
                if not self._streaming:
                    break
                    
                # Check if response is still valid
                if not self._response:
                    self.logger.warning("Response object became None during streaming")
                    break
                    
                if line:
                    try:
                        data = json.loads(line)
                        self._handle_stream_message(data)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON: {e}")
                        continue
                        
        except requests.exceptions.RequestException as e:
            if self._streaming:
                self.logger.error(f"Stream connection error: {e}")
                raise
        except Exception as e:
            if self._streaming:
                self.logger.error(f"Unexpected stream error: {e}")
                raise
        finally:
            # Clean up response object
            if self._response:
                try:
                    self._response.close()
                except:
                    pass
                self._response = None
        
    def _handle_stream_message(self, data: Dict):
        """Handle individual stream messages."""
        message_type = data.get('type', 'PRICE')
        
        if message_type == 'PRICE':
            # Update last activity time
            self._last_heartbeat = datetime.utcnow()
            
            # Emit price update
            self.price_update.emit(data)
            
        elif message_type == 'HEARTBEAT':
            # Update heartbeat time
            self._last_heartbeat = datetime.utcnow()
            
            # Emit heartbeat
            self.heartbeat.emit(data)
            
        else:
            self.logger.warning(f"Unknown message type: {message_type}")
            
    def _get_account_id(self) -> Optional[str]:
        """Get account ID from REST API."""
        try:
            # Use existing OandaAPI to get account info
            api = OandaAPI(self.api_key, self.environment)
            response = api._make_request('/v3/accounts')
            
            if response and 'accounts' in response:
                accounts = response['accounts']
                if accounts:
                    return accounts[0]['id']
                    
        except Exception as e:
            self.logger.error(f"Failed to get account ID: {e}")
            
        return None
        
    def _handle_reconnection(self):
        """Handle reconnection logic."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self.error_occurred.emit("Max reconnection attempts reached")
            self._streaming = False
            return
            
        self._reconnect_attempts += 1
        self.connection_status.emit(f"reconnecting (attempt {self._reconnect_attempts})")
        
        # Wait before reconnecting
        time.sleep(self._reconnect_delay)


class StreamingPriceManager(QObject):
    """Manager for handling streaming price data and converting to chart-compatible format."""
    
    # Signals
    candle_update = Signal(str, str, dict)  # instrument, timeframe, candle_data
    tick_update = Signal(str, dict)         # instrument, tick_data
    
    def __init__(self):
        super().__init__()
        
        self.streaming_api = OandaStreamingAPI()
        self.streaming_api.price_update.connect(self._handle_price_update)
        self.streaming_api.heartbeat.connect(self._handle_heartbeat)
        self.streaming_api.connection_status.connect(self._handle_connection_status)
        self.streaming_api.error_occurred.connect(self._handle_error)
        
        # Price tracking for candle building
        self._current_candles = {}  # {(instrument, timeframe): candle_data}
        self._last_prices = {}      # {instrument: price_data}
        
        # Candle building timers
        self._candle_timers = {}    # {(instrument, timeframe): QTimer}
        
    def start_streaming(self, instruments: List[str]) -> bool:
        """Start streaming for specified instruments."""
        return self.streaming_api.start_stream(instruments)
        
    def stop_streaming(self):
        """Stop streaming."""
        self.streaming_api.stop_stream()
        
        # Stop all candle timers
        for timer in self._candle_timers.values():
            timer.stop()
        self._candle_timers.clear()
        
    def subscribe_to_candles(self, instrument: str, timeframe: str):
        """Subscribe to candle updates for a specific instrument/timeframe."""
        key = (instrument, timeframe)
        
        if key not in self._candle_timers:
            # Create timer for this timeframe
            timer = QTimer()
            timer.timeout.connect(lambda: self._build_candle(instrument, timeframe))
            
            # Set timer interval based on timeframe
            interval_ms = self._get_candle_interval_ms(timeframe)
            timer.start(interval_ms)
            
            self._candle_timers[key] = timer
            
    def unsubscribe_from_candles(self, instrument: str, timeframe: str):
        """Unsubscribe from candle updates."""
        key = (instrument, timeframe)
        
        if key in self._candle_timers:
            self._candle_timers[key].stop()
            del self._candle_timers[key]
            
        if key in self._current_candles:
            del self._current_candles[key]
            
    def _handle_price_update(self, price_data: Dict):
        """Handle incoming price updates."""
        instrument = price_data.get('instrument')
        if not instrument:
            return
            
        # Store latest price
        self._last_prices[instrument] = price_data
        
        # Emit tick update
        self.tick_update.emit(instrument, price_data)
        
        # Update current candles
        self._update_current_candles(instrument, price_data)
        
    def _handle_heartbeat(self, heartbeat_data: Dict):
        """Handle heartbeat messages."""
        # Could be used for connection monitoring
        pass
        
    def _handle_connection_status(self, status: str):
        """Handle connection status changes."""
        print(f"Streaming connection status: {status}")
        
    def _handle_error(self, error_msg: str):
        """Handle streaming errors."""
        print(f"Streaming error: {error_msg}")
        
    def _update_current_candles(self, instrument: str, price_data: Dict):
        """Update current candle data with new price."""
        # Extract price (use mid-point of bid/ask)
        bids = price_data.get('bids', [])
        asks = price_data.get('asks', [])
        
        if not bids or not asks:
            return
            
        bid_price = float(bids[0]['price'])
        ask_price = float(asks[0]['price'])
        mid_price = (bid_price + ask_price) / 2
        
        timestamp = datetime.fromisoformat(price_data['time'].replace('Z', '+00:00'))
        
        # Update candles for all subscribed timeframes
        for (instr, timeframe), timer in self._candle_timers.items():
            if instr == instrument:
                key = (instrument, timeframe)
                
                if key not in self._current_candles:
                    # Initialize new candle
                    self._current_candles[key] = {
                        'open': mid_price,
                        'high': mid_price,
                        'low': mid_price,
                        'close': mid_price,
                        'volume': 0,
                        'timestamp': self._get_candle_timestamp(timestamp, timeframe)
                    }
                else:
                    # Update existing candle
                    candle = self._current_candles[key]
                    candle['high'] = max(candle['high'], mid_price)
                    candle['low'] = min(candle['low'], mid_price)
                    candle['close'] = mid_price
                    candle['volume'] += 1  # Increment tick count
                    
    def _build_candle(self, instrument: str, timeframe: str):
        """Build and emit completed candle."""
        key = (instrument, timeframe)
        
        if key in self._current_candles:
            candle_data = self._current_candles[key].copy()
            
            # Emit candle update
            self.candle_update.emit(instrument, timeframe, candle_data)
            
            # Reset candle for next period
            if instrument in self._last_prices:
                last_price_data = self._last_prices[instrument]
                bids = last_price_data.get('bids', [])
                asks = last_price_data.get('asks', [])
                
                if bids and asks:
                    bid_price = float(bids[0]['price'])
                    ask_price = float(asks[0]['price'])
                    mid_price = (bid_price + ask_price) / 2
                    
                    # Start new candle with current price
                    self._current_candles[key] = {
                        'open': mid_price,
                        'high': mid_price,
                        'low': mid_price,
                        'close': mid_price,
                        'volume': 0,
                        'timestamp': self._get_candle_timestamp(datetime.utcnow(), timeframe)
                    }
                    
    def _get_candle_interval_ms(self, timeframe: str) -> int:
        """Get candle building interval in milliseconds."""
        intervals = {
            'M1': 60 * 1000,      # 1 minute
            'M5': 5 * 60 * 1000,  # 5 minutes
            'M15': 15 * 60 * 1000, # 15 minutes
            'M30': 30 * 60 * 1000, # 30 minutes
            'H1': 60 * 60 * 1000,  # 1 hour
            'H4': 4 * 60 * 60 * 1000, # 4 hours
            'D': 24 * 60 * 60 * 1000  # 1 day
        }
        return intervals.get(timeframe, 60 * 1000)  # Default to 1 minute
        
    def _get_candle_timestamp(self, timestamp: datetime, timeframe: str) -> datetime:
        """Get aligned candle timestamp for timeframe."""
        # Align timestamp to candle boundary
        if timeframe == 'M1':
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == 'M5':
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == 'M15':
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == 'M30':
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == 'H1':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == 'H4':
            hour = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe == 'D':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp 