#!/usr/bin/env python3
"""
02_process_synthea.py

Script to process Synthea synthetic healthcare data.

This script loads raw Synthea data and processes it into a standardized format
with multi-diagnosis support for healthcare research.
"""

import sys
import logging
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from src.data_processing.synthea_processor import process_synthea_data
from config.settings import LOGS_DIR, PROCESSED_DATA_DIR

def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Synthea data processing...")
        logger.info(f"Project root: {project_root}")
        
        # Process Synthea data
        processed_df, report = process_synthea_data()
        
        # Save processing report
        report_path = PROCESSED_DATA_DIR / 'synthea_processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("SYNTHEA DATA PROCESSING COMPLETED")
        print("="*60)
        print(f"Total patients processed: {len(processed_df)}")
        print(f"Average age: {report['demographics']['avg_age']} years")
        print(f"ICU admission rate: {report['icu_statistics']['icu_admission_rate']}%")
        print(f"Mortality rate: {report['outcomes']['mortality_rate']}%")
        print(f"Average diagnoses per patient: {report['diagnoses']['avg_diagnoses_per_patient']}")
        print(f"High complexity patients: {report['complexity']['high_complexity_patients']}")
        
        print(f"\nTop diagnoses:")
        for diagnosis, count in list(report['diagnoses']['top_diagnoses'].items())[:5]:
            print(f"  {diagnosis}: {count}")
        
        print(f"\nOutput files:")
        print(f"  Processed data: {PROCESSED_DATA_DIR / 'synthea_processed.csv'}")
        print(f"  Report: {report_path}")
        
        logger.info("Synthea processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error processing Synthea data: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check the logs for more details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)