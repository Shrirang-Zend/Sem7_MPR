"""
target_distribution.py

Clinical literature-based target distribution for multi-diagnosis healthcare data.

This module defines the target prevalence rates based on extensive clinical
literature review and ICU patient studies.
"""

# Clinical literature-based target distribution (5,000 patients)
TARGET_DISTRIBUTION = {
    # Single conditions (40% of dataset = 2,000 patients)
    ('Diabetes',): 0.08,                      # 400 patients - 8%
    ('Cardiovascular',): 0.12,                # 600 patients - 12%
    ('Hypertension',): 0.05,                  # 250 patients - 5%
    ('Respiratory',): 0.07,                   # 350 patients - 7%
    ('Renal',): 0.03,                         # 150 patients - 3%
    ('Sepsis',): 0.05,                        # 250 patients - 5%
    
    # Two-condition combinations (35% of dataset = 1,750 patients)
    ('Diabetes', 'Hypertension'): 0.09,      # 450 patients - 9%
    ('Diabetes', 'Renal'): 0.05,             # 250 patients - 5%
    ('Diabetes', 'Cardiovascular'): 0.06,    # 300 patients - 6%
    ('Cardiovascular', 'Hypertension'): 0.08, # 400 patients - 8%
    ('Hypertension', 'Renal'): 0.04,         # 200 patients - 4%
    ('Respiratory', 'Cardiovascular'): 0.03, # 150 patients - 3%
    
    # Three+ condition combinations (20% of dataset = 1,000 patients)
    ('Diabetes', 'Hypertension', 'Renal'): 0.06,        # 300 patients - 6%
    ('Diabetes', 'Cardiovascular', 'Hypertension'): 0.05, # 250 patients - 5%
    ('Cardiovascular', 'Hypertension', 'Renal'): 0.03,   # 150 patients - 3%
    ('Respiratory', 'Cardiovascular', 'Hypertension'): 0.02, # 100 patients - 2%
    ('Sepsis', 'Cardiovascular'): 0.02,      # 100 patients - 2%
    ('Cancer', 'Sepsis'): 0.02,              # 100 patients - 2%
    
    # Complex cases (5% of dataset = 250 patients)
    ('Other',): 0.05,                         # 250 patients - 5%
}

# Clinical co-occurrence rules based on literature
# These rules guide the probabilistic generation of multi-morbidity patterns
CO_OCCURRENCE_RULES = {
    'Diabetes': {
        'Hypertension': 0.55,      # 55% of diabetes patients have hypertension
        'Renal': 0.25,             # 25% of diabetes patients have renal disease
        'Cardiovascular': 0.35,    # 35% of diabetes patients have CVD
        'Sepsis': 0.15             # 15% higher sepsis risk
    },
    'Hypertension': {
        'Renal': 0.75,             # 75% of CKD patients have hypertension
        'Cardiovascular': 0.60,    # 60% co-occurrence
        'Diabetes': 0.30           # 30% of hypertensive patients diabetic
    },
    'Cardiovascular': {
        'Hypertension': 0.70,      # 70% of CVD patients hypertensive
        'Diabetes': 0.25,          # 25% diabetic
        'Renal': 0.40,             # 40% have some renal involvement
        'Respiratory': 0.20        # 20% have respiratory comorbidities
    },
    'Respiratory': {
        'Cardiovascular': 0.30,    # 30% of COPD patients have heart disease
        'Hypertension': 0.56,      # 56% of COPD patients hypertensive
        'Sepsis': 0.12             # 12% develop sepsis
    },
    'Renal': {
        'Hypertension': 0.80,      # 80% of CKD patients hypertensive
        'Diabetes': 0.40,          # 40% diabetic nephropathy
        'Cardiovascular': 0.50     # 50% have CVD
    },
    'Sepsis': {
        'Cardiovascular': 0.30,    # 30% have underlying CVD
        'Respiratory': 0.25,       # 25% respiratory source
        'Cancer': 0.20,            # 20% cancer-related sepsis
        'Renal': 0.15              # 15% renal involvement
    },
    'Cancer': {
        'Sepsis': 0.30,            # 30% of cancer ICU patients have sepsis
        'Cardiovascular': 0.15,    # 15% have CVD complications
        'Respiratory': 0.10        # 10% respiratory involvement
    }
}

# Age-based prevalence adjustments
AGE_PREVALENCE_MODIFIERS = {
    'young_adult': {  # 18-39 years
        'Diabetes': 0.5,           # Lower diabetes prevalence
        'Hypertension': 0.3,       # Much lower hypertension
        'Cardiovascular': 0.2,     # Lower CVD
        'Trauma': 2.0,             # Higher trauma risk
        'Sepsis': 0.7              # Lower sepsis risk
    },
    'middle_aged': {  # 40-64 years
        'Diabetes': 1.0,           # Baseline prevalence
        'Hypertension': 1.0,       # Baseline
        'Cardiovascular': 1.0,     # Baseline
        'Cancer': 1.5,             # Higher cancer risk
        'Respiratory': 1.2         # Higher COPD
    },
    'elderly': {      # 65-79 years
        'Diabetes': 1.3,           # Higher diabetes
        'Hypertension': 1.5,       # Much higher hypertension
        'Cardiovascular': 1.8,     # Higher CVD
        'Renal': 1.6,              # Higher CKD
        'Sepsis': 1.4              # Higher sepsis risk
    },
    'very_elderly': { # 80+ years
        'Diabetes': 1.2,           # Slightly lower (survivor bias)
        'Hypertension': 1.8,       # Very high hypertension
        'Cardiovascular': 2.0,     # Very high CVD
        'Renal': 2.0,              # Very high CKD
        'Sepsis': 1.8,             # High sepsis risk
        'Neurological': 2.5        # Much higher stroke/dementia
    }
}

# Severity scoring based on diagnosis combinations
COMORBIDITY_SCORES = {
    1: 1.0,   # Single diagnosis
    2: 2.5,   # Two diagnoses
    3: 4.0,   # Three diagnoses
    4: 6.0,   # Four diagnoses
    5: 8.0    # Five or more diagnoses
}

# ICU-specific adjustments
ICU_PREVALENCE_ADJUSTMENTS = {
    'Sepsis': 1.5,              # 50% higher in ICU
    'Respiratory': 1.3,         # 30% higher in ICU
    'Cardiovascular': 1.2,      # 20% higher in ICU
    'Trauma': 0.8,              # 20% lower (many trauma patients bypass ICU)
    'Neurological': 1.4         # 40% higher in ICU
}

# Expected mortality rates by diagnosis (for validation)
EXPECTED_MORTALITY_RATES = {
    'Sepsis': 0.26,             # 26% ICU mortality
    'Cancer': 0.48,             # 48% ICU mortality with sepsis
    'Respiratory': 0.12,        # 12% for COPD exacerbations
    'Cardiovascular': 0.09,     # 9% for cardiac ICU
    'Renal': 0.15,              # 15% for CKD with complications
    'Trauma': 0.08,             # 8% trauma mortality
    'Neurological': 0.18,       # 18% for stroke/brain injury
    'Diabetes': 0.07,           # 7% for diabetes complications
    'Hypertension': 0.05,       # 5% for hypertensive crisis
    'Other': 0.10               # 10% baseline
}

# Length of stay patterns by diagnosis
EXPECTED_LOS_PATTERNS = {
    'ICU_LOS': {
        'Sepsis': (3, 8),           # 3-8 days average
        'Cardiovascular': (2, 5),   # 2-5 days
        'Respiratory': (4, 10),     # 4-10 days
        'Trauma': (2, 7),           # 2-7 days
        'Neurological': (5, 12),    # 5-12 days
        'Cancer': (3, 9),           # 3-9 days
        'Renal': (3, 6),            # 3-6 days
        'Other': (2, 5)             # 2-5 days
    },
    'HOSPITAL_LOS': {
        'Sepsis': (7, 15),          # 7-15 days
        'Cardiovascular': (4, 8),   # 4-8 days
        'Respiratory': (6, 12),     # 6-12 days
        'Trauma': (5, 14),          # 5-14 days
        'Neurological': (8, 20),    # 8-20 days
        'Cancer': (8, 18),          # 8-18 days
        'Renal': (5, 10),           # 5-10 days
        'Other': (3, 7)             # 3-7 days
    }
}

# Procedure and medication patterns
EXPECTED_INTERVENTIONS = {
    'HIGH_COMPLEXITY_CONDITIONS': ['Sepsis', 'Cancer', 'Neurological', 'Trauma'],
    'MECHANICAL_VENTILATION_RISK': {
        'Sepsis': 0.60,             # 60% require ventilation
        'Respiratory': 0.45,        # 45% require ventilation
        'Neurological': 0.35,       # 35% require ventilation
        'Trauma': 0.30,             # 30% require ventilation
        'Cardiovascular': 0.15      # 15% require ventilation
    },
    'DIALYSIS_RISK': {
        'Renal': 0.70,              # 70% of CKD patients on dialysis
        'Sepsis': 0.25,             # 25% develop AKI requiring dialysis
        'Cardiovascular': 0.10      # 10% cardiorenal syndrome
    }
}

def validate_target_distribution():
    """Validate that target distribution sums to 1.0 (100%)."""
    total = sum(TARGET_DISTRIBUTION.values())
    if abs(total - 1.0) > 0.01:  # Allow small floating point errors
        raise ValueError(f"Target distribution sums to {total:.3f}, not 1.0")
    return True

def get_diagnosis_prevalence(diagnosis: str) -> float:
    """Get total prevalence of a specific diagnosis across all combinations."""
    total_prevalence = 0.0
    for combo, prevalence in TARGET_DISTRIBUTION.items():
        if diagnosis in combo:
            total_prevalence += prevalence
    return total_prevalence

def get_combination_prevalence(diagnoses: tuple) -> float:
    """Get prevalence of a specific diagnosis combination."""
    # Sort diagnoses to ensure consistent lookup
    sorted_diagnoses = tuple(sorted(diagnoses))
    return TARGET_DISTRIBUTION.get(sorted_diagnoses, 0.0)

# Validate distribution on import
validate_target_distribution()