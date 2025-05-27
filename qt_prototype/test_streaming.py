"""Test script for Oanda streaming API functionality.

This script demonstrates how to use the streaming API to receive
real-time price updates from Oanda.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QLabel, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QTimer

from qt_prototype.oanda_streaming import OandaStreamingAPI, StreamingPriceManager


class StreamingTestWindow(QMainWindow):
    """Test window for streaming API functionality."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oanda Streaming API Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.start_streaming)
        button_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.stop_streaming)
        self.disconnect_btn.setEnabled(False)
        button_layout.addWidget(self.disconnect_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Price display
        self.price_label = QLabel("Price: --")
        self.price_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.price_label)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        # Detailed price info
        self.detail_text = QTextEdit()
        layout.addWidget(self.detail_text)
        
        # Initialize streaming
        self.streaming_api = OandaStreamingAPI()
        self.streaming_api.price_update.connect(self.on_price_update)
        self.streaming_api.heartbeat.connect(self.on_heartbeat)
        self.streaming_api.connection_status.connect(self.on_connection_status)
        self.streaming_api.error_occurred.connect(self.on_error)
        
        # Price tracking
        self.price_count = 0
        self.last_prices = {}
        
        # Update timer for statistics
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(5000)  # Update every 5 seconds
        
    def start_streaming(self):
        """Start streaming for EUR_USD and GBP_USD."""
        instruments = ['EUR_USD', 'GBP_USD']
        success = self.streaming_api.start_stream(instruments)
        
        if success:
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.log("Starting stream for: " + ", ".join(instruments))
        else:
            self.log("Failed to start stream")
            
    def stop_streaming(self):
        """Stop streaming."""
        self.streaming_api.stop_stream()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.log("Stopped streaming")
        
    def on_price_update(self, price_data):
        """Handle price updates."""
        instrument = price_data.get('instrument', 'Unknown')
        
        # Extract bid/ask prices
        bids = price_data.get('bids', [])
        asks = price_data.get('asks', [])
        
        if bids and asks:
            bid = float(bids[0]['price'])
            ask = float(asks[0]['price'])
            mid = (bid + ask) / 2
            spread = ask - bid
            
            # Store price
            self.last_prices[instrument] = {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread': spread,
                'time': price_data.get('time', '')
            }
            
            # Update display for EUR_USD
            if instrument == 'EUR_USD':
                self.price_label.setText(f"EUR/USD: {mid:.5f} (Spread: {spread*10000:.1f} pips)")
                
            self.price_count += 1
            
            # Log every 10th price update
            if self.price_count % 10 == 0:
                self.log(f"Price #{self.price_count}: {instrument} = {mid:.5f}")
                
    def on_heartbeat(self, heartbeat_data):
        """Handle heartbeat messages."""
        timestamp = heartbeat_data.get('time', '')
        self.log(f"Heartbeat: {timestamp}")
        
    def on_connection_status(self, status):
        """Handle connection status changes."""
        self.status_label.setText(f"Status: {status}")
        
        if status == "connected":
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        elif status == "connecting":
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: orange;")
        elif status.startswith("reconnecting"):
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: orange;")
        else:
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
            
        self.log(f"Connection status: {status}")
        
    def on_error(self, error_msg):
        """Handle errors."""
        self.log(f"ERROR: {error_msg}")
        
    def log(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Keep log size manageable
        if self.log_text.document().blockCount() > 100:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            
    def update_stats(self):
        """Update detailed statistics display."""
        if not self.last_prices:
            return
            
        stats_text = f"Price Updates Received: {self.price_count}\n\n"
        
        for instrument, price_info in self.last_prices.items():
            stats_text += f"{instrument}:\n"
            stats_text += f"  Bid: {price_info['bid']:.5f}\n"
            stats_text += f"  Ask: {price_info['ask']:.5f}\n"
            stats_text += f"  Mid: {price_info['mid']:.5f}\n"
            stats_text += f"  Spread: {price_info['spread']*10000:.1f} pips\n"
            stats_text += f"  Time: {price_info['time']}\n\n"
            
        self.detail_text.setPlainText(stats_text)
        
    def closeEvent(self, event):
        """Handle window close."""
        self.stop_streaming()
        super().closeEvent(event)


def main():
    """Main function to run the streaming test."""
    app = QApplication(sys.argv)
    
    # Check if API key is configured
    api_key = os.getenv('OANDA_API_KEY')
    if not api_key:
        print("ERROR: OANDA_API_KEY environment variable not set!")
        print("Please set your Oanda API key in the environment or .env file")
        return 1
        
    window = StreamingTestWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 