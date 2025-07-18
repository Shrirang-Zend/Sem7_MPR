#!/usr/bin/env python3
"""
08_run_api.py

Script to run the healthcare data generation API.

This script starts a FastAPI server that provides endpoints for generating
synthetic healthcare data using the trained CTGAN model.
"""

import sys
import logging
from pathlib import Path
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from config.settings import API_CONFIG

def main():
    """Main API execution."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Healthcare Data Generation API...")
    
    try:
        # Import the FastAPI app
        from src.api.app import app
        
        # Print startup information
        print("\n" + "="*60)
        print("HEALTHCARE DATA GENERATION API")
        print("="*60)
        print(f"🚀 Starting server on http://{API_CONFIG['host']}:{API_CONFIG['port']}")
        print(f"📊 Max generated rows: {API_CONFIG['max_generated_rows']}")
        print(f"⏱️  Request timeout: {API_CONFIG['timeout_seconds']}s")
        print("\n📖 API Documentation:")
        print(f"   Swagger UI: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
        print(f"   ReDoc: http://{API_CONFIG['host']}:{API_CONFIG['port']}/redoc")
        print("\n💡 Example queries:")
        print("   - 'Generate 100 patients with diabetes'")
        print("   - 'ICU patients with sepsis and cardiovascular disease'")
        print("   - 'Elderly patients with multiple comorbidities'")
        print("\n🛑 Press Ctrl+C to stop the server")
        print("="*60)
        
        # Start the server
        uvicorn.run(
            "src.api.app:app",
            host=API_CONFIG['host'],
            port=API_CONFIG['port'],
            reload=False,  # Set to True for development
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 API server stopped by user")
        logger.info("API server stopped by user")
        return 0
        
    except Exception as e:
        logger.error(f"Error starting API server: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("\nPlease check the logs for more details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)