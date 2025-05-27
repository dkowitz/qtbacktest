"""Oanda API integration with caching for live chart data.

This module provides functionality to fetch real-time and historical
candlestick data from Oanda's REST API with intelligent caching to
minimize API calls and improve performance.
"""

import os
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import threading
import time

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory, parent directory, and qt_prototype directory
    env_paths = [
        Path.cwd() / '.env',
        Path.cwd().parent / '.env', 
        Path(__file__).parent / '.env',
        Path(__file__).parent.parent / '.env'
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from: {env_path}")
            break
    else:
        # Try loading from default location
        load_dotenv()
        
except ImportError:
    print("python-dotenv not installed. Using system environment variables only.")
    print("Install with: pip install python-dotenv")


class OandaAPI:
    """Oanda API client with intelligent caching."""
    
    # Common currency pairs
    CURRENCY_PAIRS = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD', 'AUD_USD', 'NZD_USD',
        'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_CAD', 'EUR_AUD', 'EUR_NZD',
        'GBP_JPY', 'GBP_CHF', 'GBP_CAD', 'GBP_AUD', 'GBP_NZD',
        'CHF_JPY', 'CAD_JPY', 'AUD_JPY', 'NZD_JPY',
        'AUD_CAD', 'AUD_CHF', 'AUD_NZD',
        'CAD_CHF', 'NZD_CAD', 'NZD_CHF'
    ]
    
    # Timeframe mappings
    TIMEFRAMES = {
        'M1': {'seconds': 60, 'label': '1 Minute'},
        'M5': {'seconds': 300, 'label': '5 Minutes'},
        'M15': {'seconds': 900, 'label': '15 Minutes'},
        'M30': {'seconds': 1800, 'label': '30 Minutes'},
        'H1': {'seconds': 3600, 'label': '1 Hour'},
        'H4': {'seconds': 14400, 'label': '4 Hours'},
        'D': {'seconds': 86400, 'label': '1 Day'},
        'W': {'seconds': 604800, 'label': '1 Week'},
        'M': {'seconds': 2592000, 'label': '1 Month'}  # Approximate
    }
    
    def __init__(self, api_key: str = None, environment: str = None):
        """Initialize Oanda API client.
        
        Parameters
        ----------
        api_key : str, optional
            Oanda API key. If None, will try to read from environment variable OANDA_API_KEY
        environment : str, optional
            'practice' for demo account, 'live' for real account.
            If None, will try to read from OANDA_ENVIRONMENT (defaults to 'practice')
        """
        self.api_key = api_key or os.getenv('OANDA_API_KEY')
        self.environment = environment or os.getenv('OANDA_ENVIRONMENT', 'practice')
        
        if self.environment == 'practice':
            self.base_url = 'https://api-fxpractice.oanda.com'
        else:
            self.base_url = 'https://api-fxtrade.oanda.com'
            
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Get timeout from environment or use default
        self.timeout = int(os.getenv('OANDA_TIMEOUT', '30'))
        
        # Initialize cache
        self._init_cache()
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        self._request_lock = threading.Lock()
        
    def _init_cache(self):
        """Initialize SQLite cache database."""
        # Get cache directory from environment or use default
        cache_dir_env = os.getenv('OANDA_CACHE_DIR')
        if cache_dir_env:
            cache_dir = Path(cache_dir_env)
        else:
            cache_dir = Path.home() / '.backtest_workbench'
            
        cache_dir.mkdir(exist_ok=True)
        
        self.cache_path = cache_dir / 'oanda_cache.db'
        
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS candles (
                    instrument TEXT,
                    timeframe TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (instrument, timeframe, timestamp)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_candles_lookup 
                ON candles (instrument, timeframe, timestamp)
            ''')
            
    def _rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self._min_request_interval:
                time.sleep(self._min_request_interval - time_since_last)
                
            self._last_request_time = time.time()
            
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a rate-limited request to the Oanda API."""
        if not self.api_key:
            raise ValueError("API key is required. Set OANDA_API_KEY environment variable or pass api_key parameter.")
            
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
            
    def get_candles(self, instrument: str, timeframe: str, count: int = 500, 
                   from_time: datetime = None, to_time: datetime = None, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch candlestick data with intelligent caching.
        
        Parameters
        ----------
        instrument : str
            Currency pair (e.g., 'EUR_USD')
        timeframe : str
            Timeframe (e.g., 'M15', 'H1', 'D')
        count : int
            Number of candles to fetch (max 5000)
        from_time : datetime, optional
            Start time for historical data
        to_time : datetime, optional
            End time for historical data
        force_refresh : bool, optional
            If True, bypass cache and fetch fresh data from API
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data, indexed by timestamp
        """
        # If force_refresh is True, skip cache entirely
        if force_refresh:
            print(f"Force refresh requested for {instrument} {timeframe}")
        else:
            # Check cache first
            cached_data = self._get_cached_data(instrument, timeframe, from_time, to_time, count)
            
            # For live data requests (no specific time range), check cache freshness
            if from_time is None and to_time is None and cached_data is not None and len(cached_data) >= count:
                # Check if cached data is fresh enough for live updates
                latest_cached_time = cached_data.index.max()
                current_time = datetime.utcnow()
                
                # Get timeframe duration in seconds
                timeframe_seconds = self.TIMEFRAMES.get(timeframe, {}).get('seconds', 900)  # Default to 15 minutes
                
                # For live data, we want fresh data if:
                # 1. The latest cached candle is more than 2 timeframe periods old, OR
                # 2. For short timeframes (< 1 hour), if data is more than 30 minutes old
                # 3. For longer timeframes, if data is more than 2 hours old
                
                time_since_latest = (current_time - latest_cached_time).total_seconds()
                
                if timeframe_seconds < 3600:  # Less than 1 hour timeframe
                    max_age = max(timeframe_seconds * 2, 1800)  # At least 30 minutes, or 2 periods
                else:  # 1 hour or longer timeframes
                    max_age = max(timeframe_seconds * 2, 7200)  # At least 2 hours, or 2 periods
                
                print(f"Cache check for {instrument} {timeframe}: latest={latest_cached_time}, age={time_since_latest:.0f}s, max_age={max_age:.0f}s")
                
                if time_since_latest <= max_age:
                    # Cached data is fresh enough
                    print(f"Using cached data for {instrument} {timeframe} (fresh enough)")
                    return cached_data.tail(count)
                else:
                    print(f"Cached data too old for {instrument} {timeframe}, fetching fresh data")
                # If cached data is too old, continue to fetch fresh data
                
            # For historical data requests with specific time ranges, use cache if available
            elif (from_time is not None or to_time is not None) and cached_data is not None and len(cached_data) >= count:
                print(f"Using cached data for historical request {instrument} {timeframe}")
                return cached_data.tail(count)
            elif cached_data is None:
                print(f"No cached data found for {instrument} {timeframe}")
            else:
                print(f"Insufficient cached data for {instrument} {timeframe}: have {len(cached_data) if cached_data is not None else 0}, need {count}")
        
        # Determine time range for API request
        if from_time is None and to_time is None:
            # Get recent data
            params = {
                'count': count,
                'granularity': timeframe,
                'price': 'M'  # Mid prices
            }
        else:
            params = {
                'granularity': timeframe,
                'price': 'M'
            }
            if from_time:
                params['from'] = from_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            if to_time:
                params['to'] = to_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            if not from_time and not to_time:
                params['count'] = count
                
        # Make API request
        print(f"Making API request for {instrument} {timeframe} with params: {params}")
        endpoint = f"/v3/instruments/{instrument}/candles"
        data = self._make_request(endpoint, params)
        
        if not data or 'candles' not in data:
            print(f"API request failed for {instrument} {timeframe}")
            cached_data = self._get_cached_data(instrument, timeframe, from_time, to_time, count) if not force_refresh else None
            return cached_data.tail(count) if cached_data is not None else None
            
        # Convert to DataFrame
        df = self._parse_candles(data['candles'])
        
        if df is not None and not df.empty:
            print(f"Received {len(df)} candles for {instrument} {timeframe}, latest: {df.index.max()}")
            # Cache the new data
            self._cache_data(instrument, timeframe, df)
            
            # Combine with cached data if available and not force refresh
            if not force_refresh:
                cached_data = self._get_cached_data(instrument, timeframe, from_time, to_time, count)
                if cached_data is not None:
                    # Ensure both DataFrames have consistent timezone handling
                    if cached_data.index.tz is not None:
                        cached_data.index = cached_data.index.tz_localize(None)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                        
                    # Remove overlapping data and combine
                    combined = pd.concat([cached_data, df]).drop_duplicates().sort_index()
                    return combined.tail(count)
            
            return df.tail(count)
        else:
            print(f"No valid candles received for {instrument} {timeframe}")
                
        cached_data = self._get_cached_data(instrument, timeframe, from_time, to_time, count) if not force_refresh else None
        return cached_data.tail(count) if cached_data is not None else None
        
    def _parse_candles(self, candles_data: List[Dict]) -> Optional[pd.DataFrame]:
        """Parse candles data from API response into DataFrame."""
        if not candles_data:
            return None
            
        rows = []
        for candle in candles_data:
            if not candle.get('complete', True):
                continue  # Skip incomplete candles
                
            # Convert to timezone-naive timestamp for consistency with cached data
            timestamp = pd.to_datetime(candle['time'])
            if timestamp.tz is not None:
                timestamp = timestamp.tz_convert('UTC').tz_localize(None)
            mid = candle['mid']
            
            rows.append({
                'timestamp': timestamp,
                'Open': float(mid['o']),
                'High': float(mid['h']),
                'Low': float(mid['l']),
                'Close': float(mid['c']),
                'Volume': int(candle.get('volume', 0))
            })
            
        if not rows:
            return None
            
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    def _get_cached_data(self, instrument: str, timeframe: str, 
                        from_time: datetime = None, to_time: datetime = None, 
                        count: int = 500) -> Optional[pd.DataFrame]:
        """Retrieve cached candlestick data."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM candles 
                    WHERE instrument = ? AND timeframe = ?
                '''
                params = [instrument, timeframe]
                
                if from_time:
                    query += ' AND timestamp >= ?'
                    params.append(int(from_time.timestamp()))
                    
                if to_time:
                    query += ' AND timestamp <= ?'
                    params.append(int(to_time.timestamp()))
                    
                query += ' ORDER BY timestamp DESC'
                
                if not from_time and not to_time:
                    query += f' LIMIT {count * 2}'  # Get extra to ensure we have enough
                    
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return None
                    
                # Convert timestamp back to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Rename columns to match expected format
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                return df
                
        except Exception as e:
            print(f"Cache read error: {e}")
            return None
            
    def _cache_data(self, instrument: str, timeframe: str, df: pd.DataFrame):
        """Cache candlestick data to SQLite database."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                for timestamp, row in df.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO candles 
                        (instrument, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        instrument, timeframe, int(timestamp.timestamp()),
                        row['Open'], row['High'], row['Low'], row['Close'], row['Volume']
                    ))
                    
        except Exception as e:
            print(f"Cache write error: {e}")
            
    def get_latest_price(self, instrument: str) -> Optional[Dict]:
        """Get the latest price for an instrument."""
        endpoint = f"/v3/instruments/{instrument}/candles"
        params = {
            'count': 1,
            'granularity': 'M1',
            'price': 'M'
        }
        
        data = self._make_request(endpoint, params)
        
        if data and 'candles' in data and data['candles']:
            candle = data['candles'][0]
            mid = candle['mid']
            return {
                'time': candle['time'],
                'price': float(mid['c']),
                'bid': float(mid['c']),  # Simplified - in practice you'd get bid/ask
                'ask': float(mid['c'])
            }
            
        return None
        
    def test_connection(self) -> bool:
        """Test if the API connection is working."""
        try:
            endpoint = "/v3/accounts"
            result = self._make_request(endpoint)
            return result is not None
        except Exception:
            return False
            
    def clear_cache(self, instrument: str = None, timeframe: str = None):
        """Clear cached data."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                if instrument and timeframe:
                    conn.execute('DELETE FROM candles WHERE instrument = ? AND timeframe = ?', 
                               (instrument, timeframe))
                elif instrument:
                    conn.execute('DELETE FROM candles WHERE instrument = ?', (instrument,))
                else:
                    conn.execute('DELETE FROM candles')
        except Exception as e:
            print(f"Cache clear error: {e}")


# Global instance
_oanda_api = None

def get_oanda_api() -> OandaAPI:
    """Get the global Oanda API instance."""
    global _oanda_api
    if _oanda_api is None:
        _oanda_api = OandaAPI()
    return _oanda_api

def set_api_key(api_key: str, environment: str = 'practice'):
    """Set the API key for the global instance."""
    global _oanda_api
    _oanda_api = OandaAPI(api_key, environment) 