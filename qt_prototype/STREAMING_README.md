# Oanda Streaming API Implementation

This document explains the new streaming API implementation for real-time market data in the trading workbench.

## Overview

The streaming implementation provides real-time price updates from Oanda's streaming API, offering significant advantages over the traditional polling approach:

### **Polling vs Streaming Comparison**

| Feature | Polling (Current) | Streaming (New) |
|---------|------------------|-----------------|
| **Latency** | 5-60 seconds | Sub-second |
| **Data Freshness** | Only as fresh as last poll | Real-time updates |
| **API Efficiency** | Repeated full requests | Only sends changes |
| **Rate Limits** | Can hit limits with frequent polling | Single persistent connection |
| **Resource Usage** | High (continuous requests) | Low (event-driven) |
| **Market Closure** | Continues polling unnecessarily | Automatic handling |
| **Trading Suitability** | Limited | Professional-grade |

## Architecture

### Core Components

1. **`OandaStreamingAPI`** - Low-level streaming client
   - Handles HTTP chunked transfer encoding
   - Manages connection lifecycle and reconnection
   - Processes JSON price messages and heartbeats
   - Thread-safe with Qt signal integration

2. **`StreamingPriceManager`** - High-level price management
   - Converts tick data to candle format
   - Manages multiple instrument subscriptions
   - Provides chart-compatible data updates
   - Handles timeframe-specific candle building

3. **`EnhancedLiveChartWidget`** - Dual-mode chart widget
   - Supports both polling and streaming modes
   - Real-time price display with bid/ask spread
   - Seamless mode switching
   - Enhanced UI with connection status

## Key Features

### Real-Time Price Updates
- **Tick-level data**: Up to 4 price updates per second per instrument
- **Bid/Ask spreads**: Real-time spread monitoring in pips
- **Price change tracking**: Live P&L calculation with color coding
- **Sub-second latency**: Immediate price change notifications

### Connection Management
- **Automatic reconnection**: Handles network interruptions gracefully
- **Heartbeat monitoring**: 5-second heartbeats ensure connection health
- **Connection status**: Visual indicators for connection state
- **Error handling**: Comprehensive error reporting and recovery

### Candle Building
- **Real-time candles**: Live candle updates as prices change
- **Multiple timeframes**: Support for M1, M5, M15, M30, H1, H4, D
- **Boundary alignment**: Proper candle timestamp alignment
- **Volume tracking**: Tick count as volume proxy

## Usage

### Basic Streaming Setup

```python
from qt_prototype.oanda_streaming import OandaStreamingAPI

# Initialize streaming API
streaming_api = OandaStreamingAPI()

# Connect to signals
streaming_api.price_update.connect(handle_price_update)
streaming_api.connection_status.connect(handle_connection_status)

# Start streaming
instruments = ['EUR_USD', 'GBP_USD']
streaming_api.start_stream(instruments)
```

### Price Manager Usage

```python
from qt_prototype.oanda_streaming import StreamingPriceManager

# Initialize price manager
price_manager = StreamingPriceManager()

# Connect to candle updates
price_manager.candle_update.connect(handle_candle_update)
price_manager.tick_update.connect(handle_tick_update)

# Start streaming and subscribe to candles
price_manager.start_streaming(['EUR_USD'])
price_manager.subscribe_to_candles('EUR_USD', 'M15')
```

### Enhanced Chart Widget

```python
from qt_prototype.live_chart_streaming import EnhancedLiveChartWidget

# Create enhanced chart widget
chart_widget = EnhancedLiveChartWidget()

# Widget automatically handles mode switching between polling and streaming
# Users can toggle between modes via radio buttons in the UI
```

## Testing

### Test Script
Run the streaming test to verify functionality:

```bash
cd btest/qt_prototype
python test_streaming.py
```

The test script provides:
- Real-time price display for EUR/USD and GBP/USD
- Connection status monitoring
- Price update statistics
- Error logging and debugging

### Expected Output
When running successfully, you should see:
- Connection status: "connected" (green)
- Price updates every few seconds
- Heartbeat messages every 5 seconds
- Real-time bid/ask spreads in pips

## Configuration

### Environment Variables
```bash
OANDA_API_KEY=your_api_key_here
OANDA_ENVIRONMENT=practice  # or 'live'
OANDA_TIMEOUT=30
```

### API Endpoints
- **Practice**: `https://stream-fxpractice.oanda.com`
- **Live**: `https://stream-fxtrade.oanda.com`

## Integration with Existing System

### Backward Compatibility
The streaming implementation is designed to work alongside the existing polling system:

- **Existing charts**: Continue to work with polling mode
- **New charts**: Can use streaming mode for real-time updates
- **Mode switching**: Users can toggle between polling and streaming
- **Same data format**: Streaming produces compatible DataFrame output

### Migration Path
1. **Phase 1**: Test streaming with new enhanced chart widget
2. **Phase 2**: Add streaming option to existing live charts
3. **Phase 3**: Make streaming the default for new charts
4. **Phase 4**: Deprecate polling for live data (keep for historical)

## Benefits for Trading

### Live View Auto-Refresh
- **Immediate updates**: See price changes as they happen
- **No polling delays**: Eliminate 5-60 second refresh intervals
- **Smooth price action**: Continuous price flow for better analysis

### Order Management (Future)
- **Real-time fills**: Immediate notification of order executions
- **Position tracking**: Live P&L updates
- **Market events**: Instant notification of market conditions

### Account Monitoring (Future)
- **Balance updates**: Real-time account balance changes
- **Margin monitoring**: Live margin requirement tracking
- **Risk management**: Immediate alerts for risk thresholds

## Performance Considerations

### Resource Usage
- **Memory**: Minimal overhead for price storage
- **CPU**: Event-driven processing (low CPU usage)
- **Network**: Single persistent connection vs multiple requests
- **Battery**: Lower power consumption on mobile devices

### Scalability
- **Multiple instruments**: Single connection handles multiple pairs
- **Multiple timeframes**: Efficient candle building for all timeframes
- **Connection pooling**: Reuse connections across widgets

## Error Handling

### Connection Issues
- **Network interruption**: Automatic reconnection with exponential backoff
- **API errors**: Graceful degradation with error reporting
- **Rate limiting**: Proper handling of API rate limits
- **Timeout handling**: Configurable timeout settings

### Data Quality
- **Missing data**: Handles gaps in price streams
- **Duplicate data**: Deduplication of price updates
- **Timestamp alignment**: Proper handling of timezone issues
- **Market hours**: Automatic handling of market closures

## Future Enhancements

### Transaction Streaming
- **Order events**: Real-time order status updates
- **Trade events**: Immediate trade execution notifications
- **Account events**: Balance and position changes

### Advanced Features
- **Market depth**: Level 2 order book data
- **News integration**: Real-time news event streaming
- **Economic calendar**: Live economic event notifications
- **Custom alerts**: User-defined price and event alerts

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check API key configuration
   - Verify network connectivity
   - Ensure correct environment (practice/live)

2. **No Price Updates**
   - Check instrument subscription
   - Verify market hours
   - Check connection status

3. **High CPU Usage**
   - Reduce number of subscribed instruments
   - Increase candle building intervals
   - Check for memory leaks

### Debug Mode
Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The streaming API implementation represents a significant upgrade from polling-based data retrieval, providing:

- **Professional-grade performance** suitable for serious trading
- **Real-time responsiveness** for immediate market awareness
- **Efficient resource usage** with lower API consumption
- **Scalable architecture** supporting future trading features

This foundation enables the evolution from a back-testing tool to a comprehensive trading platform with real-time capabilities essential for live trading operations. 