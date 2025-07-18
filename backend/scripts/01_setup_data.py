#!/usr/bin/env python3
"""
01_setup_data.py

Script to setup and validate raw data for the healthcare data generation system.

This script checks for the presence of required Synthea and MIMIC-III data files
and provides guidance on obtaining missing data.
"""

import sys
import logging
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from config.settings import (
    SYNTHEA_DATA_DIR, MIMIC_BASE_DIR, RAW_DATA_DIR,
    PROJECT_ROOT, DATA_DIR
)

def check_directory_structure():
    """Check and create necessary directory structure."""
    logger = logging.getLogger(__name__)
    
    required_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        RAW_DATA_DIR / "synthea",
        RAW_DATA_DIR / "mimic-iii-clinical-database-demo-1.4",
        DATA_DIR / "processed",
        DATA_DIR / "final",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "models"
    ]
    
    logger.info("Checking directory structure...")
    
    for directory in required_dirs:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.debug(f"Directory exists: {directory}")
    
    logger.info("Directory structure validated")

def check_synthea_data():
    """Check for Synthea data availability."""
    logger = logging.getLogger(__name__)
    
    logger.info("Checking Synthea data...")
    
    # Check if Synthea base directory exists
    synthea_base = SYNTHEA_DATA_DIR.parent.parent if 'output_' in str(SYNTHEA_DATA_DIR) else SYNTHEA_DATA_DIR.parent
    if not synthea_base.exists():
        logger.warning(f"Synthea directory not found: {synthea_base}")
        return False
    
    # Look for output directories
    output_dirs = list(synthea_base.glob("output_*"))
    if not output_dirs:
        logger.warning("No Synthea output directories found")
        return False
    
    # Check the most recent output directory
    latest_output = max(output_dirs)
    csv_dir = latest_output / "csv"
    
    if not csv_dir.exists():
        logger.warning(f"CSV directory not found in: {latest_output}")
        return False
    
    # Required Synthea files
    required_files = [
        'patients.csv',
        'conditions.csv',
        'procedures.csv',
        'medications.csv',
        'encounters.csv',
        'observations.csv'
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = csv_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            logger.info(f"Found {filename}: {file_size:,} bytes")
        else:
            missing_files.append(filename)
            logger.warning(f"Missing {filename}")
    
    if missing_files:
        logger.error(f"Missing Synthea files: {missing_files}")
        return False
    
    logger.info("Synthea data validated successfully")
    return True

def check_mimic_data():
    """Check for MIMIC-III data availability."""
    logger = logging.getLogger(__name__)
    
    logger.info("Checking MIMIC-III data...")
    
    if not MIMIC_BASE_DIR.exists():
        logger.warning(f"MIMIC directory not found: {MIMIC_BASE_DIR}")
        return False
    
    # Required MIMIC files
    required_files = [
        'PATIENTS.csv',
        'ADMISSIONS.csv',
        'DIAGNOSES_ICD.csv',
        'PROCEDURES_ICD.csv',
        'PRESCRIPTIONS.csv',
        'ICUSTAYS.csv'
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = MIMIC_BASE_DIR / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            logger.info(f"Found {filename}: {file_size:,} bytes")
        else:
            missing_files.append(filename)
            logger.warning(f"Missing {filename}")
    
    if missing_files:
        logger.error(f"Missing MIMIC files: {missing_files}")
        return False
    
    logger.info("MIMIC-III data validated successfully")
    return True

def print_data_acquisition_guide():
    """Print guidance on how to obtain required data."""
    print("\n" + "="*80)
    print("DATA ACQUISITION GUIDE")
    print("="*80)
    
    print("\n📁 SYNTHEA DATA:")
    print("   Synthea generates synthetic healthcare data.")
    print("   ")
    print("   Steps to generate Synthea data:")
    print("   1. Clone Synthea repository:")
    print("      git clone https://github.com/synthetichealth/synthea.git")
    print("   2. Navigate to Synthea directory and generate data:")
    print("      cd synthea")
    print("      ./run_synthea -p 4000  # Generate 4000 patients")
    print("   3. Copy generated CSV files to your project:")
    print(f"      cp output/csv/*.csv {RAW_DATA_DIR / 'synthea'}/")
    
    print("\n📁 MIMIC-III DEMO DATA:")
    print("   MIMIC-III demo is a subset of the MIMIC-III clinical database.")
    print("   URL: https://physionet.org/content/mimiciii-demo/1.4/")
    print("   ")
    print("   Steps to download MIMIC-III demo:")
    print("   1. Download using wget (no registration required for demo):")
    print("      wget -r -N -c -np https://physionet.org/files/mimiciii-demo/1.4/")
    print("   2. Extract files to your project:")
    print(f"      cp mimiciii-demo/1.4/*.csv {MIMIC_BASE_DIR}/")
    print("   ")
    print("   Alternative: Manual download from PhysioNet website")
    
    print(f"\n📂 Expected directory structure:")
    print(f"   {RAW_DATA_DIR}/")
    print(f"   ├── synthea/")
    print(f"   │   └── output_YYYYMMDD_HHMMSS/")
    print(f"   │       └── csv/")
    print(f"   │           ├── patients.csv")
    print(f"   │           ├── conditions.csv")
    print(f"   │           ├── procedures.csv")
    print(f"   │           ├── medications.csv")
    print(f"   │           ├── encounters.csv")
    print(f"   │           └── observations.csv")
    print(f"   └── mimic-iii-clinical-database-demo-1.4/")
    print(f"       ├── PATIENTS.csv")
    print(f"       ├── ADMISSIONS.csv")
    print(f"       ├── DIAGNOSES_ICD.csv")
    print(f"       ├── PROCEDURES_ICD.csv")
    print(f"       ├── PRESCRIPTIONS.csv")
    print(f"       └── ICUSTAYS.csv")

def validate_python_environment():
    """Validate Python environment and dependencies."""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        logger.warning("Python 3.8+ recommended for best compatibility")
    
    # Check for required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'scipy',
        'matplotlib', 'seaborn', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"Package available: {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"Package missing: {package}")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        print(f"\n⚠️  Missing required packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    logger.info("Python environment validated successfully")
    return True

def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data setup validation...")
    logger.info(f"Project root: {project_root}")
    
    # Check directory structure
    check_directory_structure()
    
    # Validate Python environment
    env_valid = validate_python_environment()
    
    # Check data availability
    synthea_available = check_synthea_data()
    mimic_available = check_mimic_data()
    
    # Print results
    print("\n" + "="*60)
    print("DATA SETUP VALIDATION RESULTS")
    print("="*60)
    
    print(f"✅ Directory structure: Created/Validated")
    print(f"{'✅' if env_valid else '❌'} Python environment: {'Valid' if env_valid else 'Issues found'}")
    print(f"{'✅' if synthea_available else '❌'} Synthea data: {'Available' if synthea_available else 'Missing'}")
    print(f"{'✅' if mimic_available else '❌'} MIMIC-III data: {'Available' if mimic_available else 'Missing'}")
    
    if not synthea_available or not mimic_available:
        print_data_acquisition_guide()
        
        print(f"\n⚠️  Next steps:")
        if not synthea_available:
            print(f"   1. Generate or download Synthea data")
        if not mimic_available:
            print(f"   2. Download MIMIC-III demo data")
        print(f"   3. Run this script again to validate")
        print(f"   4. Proceed with data processing scripts")
        
        return 1
    
    print(f"\n🎉 Setup complete! Ready to proceed with data processing.")
    print(f"\nNext steps:")
    print(f"   1. python scripts/02_process_synthea.py")
    print(f"   2. python scripts/03_process_mimic.py") 
    print(f"   3. python scripts/04_combine_datasets.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)