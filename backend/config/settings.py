"""
settings.py

Configuration settings for the healthcare data generation system.

This module contains all configuration constants, paths, and settings
used throughout the project.
"""

import os
from pathlib import Path

# Base project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Synthea data paths
SYNTHEA_BASE_DIR = RAW_DATA_DIR / "synthea"
# Auto-detect the latest Synthea output directory
SYNTHEA_OUTPUT_DIRS = list(SYNTHEA_BASE_DIR.glob("output_*")) if SYNTHEA_BASE_DIR.exists() else []
if SYNTHEA_OUTPUT_DIRS:
    SYNTHEA_DATA_DIR = max(SYNTHEA_OUTPUT_DIRS) / "csv"  # Get the most recent output
else:
    SYNTHEA_DATA_DIR = SYNTHEA_BASE_DIR / "csv"  # Fallback

# MIMIC-III data paths
MIMIC_BASE_DIR = RAW_DATA_DIR / "mimic-iii-clinical-database-demo-1.4"

# Final dataset configuration
TARGET_DATASET_SIZE = 5000
SYNTHEA_TARGET_SIZE = 4000  # 80% of final dataset
MIMIC_TARGET_SIZE = 100     # ~100 patients from demo (remaining 20%)

# Disease categories mapping
DISEASE_CATEGORIES = {
    'DIABETES': ['diabetes', 'diabetic', 'hyperglycemia', 'insulin', 'glucose'],
    'CARDIOVASCULAR': ['cardiac', 'heart', 'coronary', 'myocardial', 'cardiovascular', 'angina'],
    'HYPERTENSION': ['hypertension', 'hypertensive', 'blood pressure', 'high bp'],
    'RENAL': ['kidney', 'renal', 'nephritis', 'dialysis', 'creatinine'],
    'RESPIRATORY': ['copd', 'asthma', 'pneumonia', 'respiratory', 'lung', 'bronchitis'],
    'SEPSIS': ['sepsis', 'septic', 'infection', 'bacteremia'],
    'NEUROLOGICAL': ['stroke', 'seizure', 'neurological', 'brain', 'cerebral'],
    'TRAUMA': ['trauma', 'fracture', 'injury', 'accident', 'wound'],
    'CANCER': ['cancer', 'tumor', 'malignant', 'carcinoma', 'oncology'],
    'OTHER': ['other', 'unspecified', 'unknown']
}

# ICD-9 to category mapping (for MIMIC data)
ICD9_CATEGORY_MAPPING = {
    # Diabetes
    '250': 'DIABETES',
    # Cardiovascular
    '410': 'CARDIOVASCULAR', '411': 'CARDIOVASCULAR', '412': 'CARDIOVASCULAR',
    '413': 'CARDIOVASCULAR', '414': 'CARDIOVASCULAR', '428': 'CARDIOVASCULAR',
    # Hypertension
    '401': 'HYPERTENSION', '402': 'HYPERTENSION', '403': 'HYPERTENSION',
    '404': 'HYPERTENSION', '405': 'HYPERTENSION',
    # Renal
    '580': 'RENAL', '581': 'RENAL', '582': 'RENAL', '583': 'RENAL',
    '584': 'RENAL', '585': 'RENAL', '586': 'RENAL',
    # Respiratory
    '490': 'RESPIRATORY', '491': 'RESPIRATORY', '492': 'RESPIRATORY',
    '493': 'RESPIRATORY', '494': 'RESPIRATORY', '496': 'RESPIRATORY',
    '486': 'RESPIRATORY',
    # Sepsis
    '038': 'SEPSIS', '995': 'SEPSIS',
    # Neurological
    '430': 'NEUROLOGICAL', '431': 'NEUROLOGICAL', '432': 'NEUROLOGICAL',
    '433': 'NEUROLOGICAL', '434': 'NEUROLOGICAL', '435': 'NEUROLOGICAL',
    '436': 'NEUROLOGICAL', '437': 'NEUROLOGICAL', '345': 'NEUROLOGICAL',
    # Trauma
    '800': 'TRAUMA', '801': 'TRAUMA', '802': 'TRAUMA', '803': 'TRAUMA',
    '804': 'TRAUMA', '805': 'TRAUMA', '806': 'TRAUMA', '807': 'TRAUMA',
    '808': 'TRAUMA', '809': 'TRAUMA', '810': 'TRAUMA', '811': 'TRAUMA',
    '812': 'TRAUMA', '813': 'TRAUMA', '814': 'TRAUMA', '815': 'TRAUMA',
    '816': 'TRAUMA', '817': 'TRAUMA', '818': 'TRAUMA', '819': 'TRAUMA',
    # Cancer
    '140': 'CANCER', '141': 'CANCER', '142': 'CANCER', '143': 'CANCER',
    '144': 'CANCER', '145': 'CANCER', '146': 'CANCER', '147': 'CANCER',
    '148': 'CANCER', '149': 'CANCER', '150': 'CANCER', '151': 'CANCER',
    '152': 'CANCER', '153': 'CANCER', '154': 'CANCER', '155': 'CANCER',
    '156': 'CANCER', '157': 'CANCER', '158': 'CANCER', '159': 'CANCER',
    '160': 'CANCER', '161': 'CANCER', '162': 'CANCER', '163': 'CANCER',
    '164': 'CANCER', '165': 'CANCER', '170': 'CANCER', '171': 'CANCER',
    '172': 'CANCER', '173': 'CANCER', '174': 'CANCER', '175': 'CANCER',
    '176': 'CANCER', '179': 'CANCER', '180': 'CANCER', '181': 'CANCER',
    '182': 'CANCER', '183': 'CANCER', '184': 'CANCER', '185': 'CANCER',
    '186': 'CANCER', '187': 'CANCER', '188': 'CANCER', '189': 'CANCER',
    '190': 'CANCER', '191': 'CANCER', '192': 'CANCER', '193': 'CANCER',
    '194': 'CANCER', '195': 'CANCER', '196': 'CANCER', '197': 'CANCER',
    '198': 'CANCER', '199': 'CANCER'
}

# Data processing parameters
RANDOM_SEED = 42
MIN_AGE = 18  # Only adult patients
MAX_AGE = 100

# ICU stay parameters
MIN_ICU_LOS_HOURS = 24  # Minimum 24 hours for ICU stay
MAX_ICU_LOS_DAYS = 30   # Maximum 30 days

# Hospital stay parameters
MIN_HOSPITAL_LOS_DAYS = 1
MAX_HOSPITAL_LOS_DAYS = 60

# Age groups
AGE_GROUPS = {
    'young_adult': (18, 39),
    'middle_aged': (40, 64),
    'elderly': (65, 79),
    'very_elderly': (80, 100)
}

# Risk levels
RISK_LEVELS = ['low', 'medium', 'high', 'critical']

# Utilization categories
UTILIZATION_CATEGORIES = ['low', 'medium', 'high']

# CTGAN training parameters - optimized for better performance
CTGAN_PARAMS = {
    'epochs': 800,  # Balanced epochs for faster iteration
    'batch_size': 250,  # Reduced batch size for stability
    'generator_dim': (1024, 512, 256),  # Larger generator for better diversity
    'discriminator_dim': (512, 256, 128),  # Larger discriminator to match generator
    'pac': 10,  # PAC parameter for better mode coverage
    'lr': 5e-5,  # Reduced learning rate for stability
    'decay': 1e-6  # Smaller decay rate
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'literature_alignment': 0.90,      # 90% of prevalence rates within published ranges
    'diversity_score': 0.15,           # No single combination >15% of dataset
    'clinical_validity': 0.95,         # 95% of records pass clinical logic checks
    'statistical_quality': 0.05,       # KS-test p-values >0.05
    'overall_quality': 0.85            # Overall validation score ≥85%
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'max_generated_rows': 2000,
    'timeout_seconds': 300
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'healthcare_system.log',
            'mode': 'a',
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# File paths for outputs
OUTPUT_FILES = {
    'synthea_processed': PROCESSED_DATA_DIR / 'synthea_processed.csv',
    'mimic_processed': PROCESSED_DATA_DIR / 'mimic_processed.csv',
    'combined_dataset': PROCESSED_DATA_DIR / 'combined_dataset.csv',
    'final_dataset': FINAL_DATA_DIR / 'healthcare_dataset_multi_diagnoses.csv',
    'ctgan_model': MODELS_DIR / 'ctgan_model.pkl',
    'validation_report': FINAL_DATA_DIR / 'validation_report.json'
}

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FINAL_DATA_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)