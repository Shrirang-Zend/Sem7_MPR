#!/usr/bin/env python3
"""
05_validate_dataset.py

Script to validate the final healthcare dataset.

This script runs comprehensive validation checks on the generated dataset
to ensure clinical validity, statistical quality, and distribution alignment.
"""

import sys
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from config.settings import FINAL_DATA_DIR, LOGS_DIR
from config.target_distribution import TARGET_DISTRIBUTION, validate_target_distribution

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def load_final_dataset():
    """Load the final generated dataset."""
    dataset_path = FINAL_DATA_DIR / 'healthcare_dataset_multi_diagnoses.csv'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded final dataset: {len(df)} patients")
    return df

def validate_basic_structure(df):
    """Validate basic dataset structure and completeness."""
    logger.info("Validating basic dataset structure...")
    
    required_columns = [
        'patient_id', 'diagnoses', 'age', 'gender', 'icu_los_days',
        'hospital_los_days', 'mortality', 'comorbidity_score', 'risk_level'
    ]
    
    validation_results = {
        'total_patients': len(df),
        'total_columns': len(df.columns),
        'missing_columns': [],
        'data_quality': {}
    }
    
    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            validation_results['missing_columns'].append(col)
    
    # Check data quality
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = missing_count / len(df) * 100
        
        validation_results['data_quality'][col] = {
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_pct, 2)
        }
    
    # Overall quality assessment
    avg_missing = sum(df.isnull().sum()) / (len(df) * len(df.columns)) * 100
    validation_results['overall_missing_percentage'] = round(avg_missing, 2)
    
    return validation_results

def validate_clinical_patterns(df):
    """Validate clinical patterns and logical consistency."""
    logger.info("Validating clinical patterns...")
    
    validation_results = {
        'age_distribution': {},
        'clinical_logic_checks': {},
        'diagnosis_patterns': {}
    }
    
    # Age distribution
    validation_results['age_distribution'] = {
        'min_age': int(df['age'].min()),
        'max_age': int(df['age'].max()),
        'mean_age': round(df['age'].mean(), 1),
        'adults_only': (df['age'] >= 18).all()
    }
    
    # Clinical logic checks
    icu_patients = df[df['has_icu_stay'] == 1]
    validation_results['clinical_logic_checks'] = {
        'icu_los_consistency': (icu_patients['icu_los_days'] > 0).all(),
        'hospital_icu_relationship': (df['hospital_los_days'] >= df['icu_los_days']).all(),
        'mortality_age_correlation': df.groupby('mortality')['age'].mean().to_dict(),
        'high_complexity_patients': int(df['high_complexity'].sum())
    }
    
    # Diagnosis patterns
    all_diagnoses = []
    for diagnoses_json in df['diagnoses']:
        try:
            diagnoses = json.loads(diagnoses_json)
            all_diagnoses.extend(diagnoses)
        except:
            continue
    
    diagnosis_counts = pd.Series(all_diagnoses).value_counts()
    validation_results['diagnosis_patterns'] = {
        'unique_diagnoses': len(diagnosis_counts),
        'top_diagnoses': diagnosis_counts.head(10).to_dict(),
        'avg_diagnoses_per_patient': round(df['num_diagnoses'].mean(), 1)
    }
    
    return validation_results

def validate_distribution_alignment(df):
    """Validate alignment with target clinical distributions."""
    logger.info("Validating distribution alignment...")
    
    from collections import Counter
    
    # Calculate current distribution
    current_distribution = {}
    combination_counts = Counter()
    
    for diagnoses_json in df['diagnoses']:
        try:
            diagnoses = json.loads(diagnoses_json)
            if diagnoses:
                combination = tuple(sorted(diagnoses))
                combination_counts[combination] += 1
        except:
            continue
    
    total_patients = len(df)
    for combination, count in combination_counts.items():
        current_distribution[combination] = count / total_patients
    
    # Compare with targets
    validation_results = {
        'target_vs_actual': {},
        'major_deviations': [],
        'overall_alignment': {}
    }
    
    deviations = []
    for target_combo, target_pct in TARGET_DISTRIBUTION.items():
        actual_pct = current_distribution.get(target_combo, 0.0)
        difference = abs(actual_pct - target_pct)
        
        validation_results['target_vs_actual'][str(target_combo)] = {
            'target_percentage': round(target_pct * 100, 1),
            'actual_percentage': round(actual_pct * 100, 1),
            'difference': round(difference * 100, 1)
        }
        
        if difference > 0.05:  # >5% deviation
            deviations.append({
                'combination': str(target_combo),
                'difference': round(difference * 100, 1)
            })
    
    validation_results['major_deviations'] = deviations
    
    # Overall alignment metrics
    avg_deviation = sum(abs(current_distribution.get(combo, 0) - target_pct) 
                       for combo, target_pct in TARGET_DISTRIBUTION.items()) / len(TARGET_DISTRIBUTION)
    
    validation_results['overall_alignment'] = {
        'average_deviation_percentage': round(avg_deviation * 100, 1),
        'quality_score': 'excellent' if avg_deviation < 0.02 else 'good' if avg_deviation < 0.05 else 'needs_improvement'
    }
    
    return validation_results

def validate_icu_patterns(df):
    """Validate ICU-specific patterns and outcomes."""
    logger.info("Validating ICU patterns...")
    
    icu_patients = df[df['has_icu_stay'] == 1]
    non_icu_patients = df[df['has_icu_stay'] == 0]
    
    validation_results = {
        'icu_statistics': {
            'total_icu_patients': len(icu_patients),
            'icu_admission_rate': round(len(icu_patients) / len(df) * 100, 1),
            'average_icu_los': round(icu_patients['icu_los_days'].mean(), 1) if len(icu_patients) > 0 else 0,
            'icu_mortality_rate': round(icu_patients['mortality'].mean() * 100, 1) if len(icu_patients) > 0 else 0,
            'non_icu_mortality_rate': round(non_icu_patients['mortality'].mean() * 100, 1) if len(non_icu_patients) > 0 else 0
        },
        'complexity_patterns': {
            'high_complexity_in_icu': int(icu_patients['high_complexity'].sum()) if len(icu_patients) > 0 else 0,
            'avg_comorbidity_score_icu': round(icu_patients['comorbidity_score'].mean(), 1) if len(icu_patients) > 0 else 0,
            'avg_comorbidity_score_non_icu': round(non_icu_patients['comorbidity_score'].mean(), 1) if len(non_icu_patients) > 0 else 0
        }
    }
    
    return validation_results

def generate_validation_report(df):
    """Generate comprehensive validation report."""
    logger.info("Generating comprehensive validation report...")
    
    report = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'dataset_summary': {
            'total_patients': len(df),
            'data_sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'file_size_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
        }
    }
    
    # Run all validations
    report['basic_structure'] = validate_basic_structure(df)
    report['clinical_patterns'] = validate_clinical_patterns(df)
    report['distribution_alignment'] = validate_distribution_alignment(df)
    report['icu_patterns'] = validate_icu_patterns(df)
    
    # Overall quality score
    missing_pct = report['basic_structure']['overall_missing_percentage']
    alignment_quality = report['distribution_alignment']['overall_alignment']['quality_score']
    clinical_checks = report['clinical_patterns']['clinical_logic_checks']
    
    quality_issues = []
    if missing_pct > 5:
        quality_issues.append('high_missing_data')
    if alignment_quality == 'needs_improvement':
        quality_issues.append('poor_distribution_alignment')
    if not clinical_checks.get('icu_los_consistency', True):
        quality_issues.append('clinical_logic_issues')
    
    report['overall_quality'] = {
        'status': 'excellent' if not quality_issues else 'good' if len(quality_issues) <= 1 else 'needs_review',
        'issues': quality_issues,
        'ready_for_production': len(quality_issues) <= 1
    }
    
    return report

def main():
    """Main validation execution."""
    global logger
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting dataset validation...")
        
        # Load dataset
        df = load_final_dataset()
        
        # Generate validation report
        validation_report = generate_validation_report(df)
        
        # Save validation report
        report_path = FINAL_DATA_DIR / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=convert_numpy_types)
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET VALIDATION COMPLETED")
        print("="*60)
        print(f"Total patients: {validation_report['dataset_summary']['total_patients']}")
        print(f"Overall quality: {validation_report['overall_quality']['status'].upper()}")
        print(f"Ready for production: {'✅ YES' if validation_report['overall_quality']['ready_for_production'] else '⚠️  NEEDS REVIEW'}")
        
        # Basic statistics
        basic = validation_report['basic_structure']
        print(f"\nData Quality:")
        print(f"  Missing data: {basic['overall_missing_percentage']}%")
        print(f"  Complete columns: {len(basic['data_quality']) - len([k for k, v in basic['data_quality'].items() if v['missing_count'] > 0])}/{len(basic['data_quality'])}")
        
        # Clinical patterns
        clinical = validation_report['clinical_patterns']
        print(f"\nClinical Patterns:")
        print(f"  Age range: {clinical['age_distribution']['min_age']}-{clinical['age_distribution']['max_age']} years")
        print(f"  Average diagnoses: {clinical['diagnosis_patterns']['avg_diagnoses_per_patient']}")
        print(f"  Unique diagnoses: {clinical['diagnosis_patterns']['unique_diagnoses']}")
        
        # ICU statistics
        icu = validation_report['icu_patterns']
        print(f"\nICU Statistics:")
        print(f"  ICU admission rate: {icu['icu_statistics']['icu_admission_rate']}%")
        print(f"  ICU mortality rate: {icu['icu_statistics']['icu_mortality_rate']}%")
        print(f"  Average ICU LOS: {icu['icu_statistics']['average_icu_los']} days")
        
        # Distribution alignment
        alignment = validation_report['distribution_alignment']
        print(f"\nDistribution Alignment:")
        print(f"  Quality score: {alignment['overall_alignment']['quality_score'].upper()}")
        print(f"  Average deviation: {alignment['overall_alignment']['average_deviation_percentage']}%")
        
        if alignment['major_deviations']:
            print(f"  Major deviations: {len(alignment['major_deviations'])}")
            for dev in alignment['major_deviations'][:3]:
                print(f"    {dev['combination']}: {dev['difference']}% off")
        
        # Quality issues
        if validation_report['overall_quality']['issues']:
            print(f"\n⚠️  Quality Issues:")
            for issue in validation_report['overall_quality']['issues']:
                print(f"  - {issue.replace('_', ' ').title()}")
        
        print(f"\nValidation report saved to: {report_path}")
        
        logger.info("Dataset validation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)