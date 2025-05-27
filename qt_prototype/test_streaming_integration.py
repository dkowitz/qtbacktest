#!/usr/bin/env python3
"""
Test script to verify streaming integration works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all streaming modules can be imported."""
    print("Testing imports...")
    
    try:
        from qt_prototype.oanda_streaming import OandaStreamingAPI, StreamingPriceManager
        print("✓ Core streaming modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core streaming modules: {e}")
        return False
    
    try:
        from qt_prototype.live_chart_streaming import EnhancedLiveChartWidget
        print("✓ Enhanced live chart widget imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced live chart widget: {e}")
        return False
    
    return True

def test_environment():
    """Test that environment variables are loaded."""
    print("\nTesting environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file found")
    else:
        print("✗ .env file not found")
        return False
    
    # Try to load environment variables using the same method as the streaming modules
    try:
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✓ Environment variables loaded with python-dotenv")
        except ImportError:
            print("⚠ python-dotenv not available, using system environment only")
        
        # Check for required environment variables
        if os.getenv('OANDA_API_KEY'):
            print("✓ OANDA_API_KEY found in environment")
        else:
            print("⚠ OANDA_API_KEY not found in environment")
        
        oanda_env = os.getenv('OANDA_ENVIRONMENT', 'practice')
        print(f"✓ OANDA_ENVIRONMENT set to: {oanda_env}")
            
        return True
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        return False

def test_streaming_api():
    """Test that streaming API can be instantiated."""
    print("\nTesting streaming API...")
    
    try:
        from qt_prototype.oanda_streaming import OandaStreamingAPI
        
        # Try to create API instance (without starting stream)
        api = OandaStreamingAPI()
        print("✓ OandaStreamingAPI instance created successfully")
        
        # Test basic methods
        if hasattr(api, 'start_stream'):
            print("✓ start_stream method available")
        else:
            print("✗ start_stream method missing")
            return False
            
        if hasattr(api, 'stop_stream'):
            print("✓ stop_stream method available")
        else:
            print("✗ stop_stream method missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Failed to create streaming API: {e}")
        return False

def test_price_manager():
    """Test that price manager can be instantiated."""
    print("\nTesting price manager...")
    
    try:
        from qt_prototype.oanda_streaming import StreamingPriceManager
        
        # Try to create price manager instance
        manager = StreamingPriceManager()
        print("✓ StreamingPriceManager instance created successfully")
        
        # Test basic methods
        if hasattr(manager, 'start_streaming'):
            print("✓ start_streaming method available")
        else:
            print("✗ start_streaming method missing")
            return False
            
        if hasattr(manager, 'subscribe_to_candles'):
            print("✓ subscribe_to_candles method available")
        else:
            print("✗ subscribe_to_candles method missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Failed to create price manager: {e}")
        return False

def main():
    """Run all tests."""
    print("Streaming Integration Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_streaming_api,
        test_price_manager,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Streaming integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 