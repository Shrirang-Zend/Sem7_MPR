#!/usr/bin/env python3
"""
04_combine_datasets.py

Script to combine Synthea and MIMIC-III datasets.

This script combines the processed Synthea and MIMIC-III data according to
clinical literature-based target distributions to create the final balanced dataset.
"""

import sys
import logging
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from src.data_processing.data_combiner import combine_datasets
from config.settings import LOGS_DIR, FINAL_DATA_DIR

def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting dataset combination...")
        logger.info(f"Project root: {project_root}")
        
        # Combine datasets
        final_df, report = combine_datasets()
        
        # Save combination report
        report_path = FINAL_DATA_DIR / 'combination_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET COMBINATION COMPLETED")
        print("="*60)
        print(f"Final dataset size: {len(final_df)} patients")
        print(f"Target dataset size: {report['dataset_size']}")
        
        # Source distribution
        print(f"\nSource distribution:")
        for source, percentage in report['source_distribution'].items():
            print(f"  {source}: {percentage:.1%}")
        
        # Demographics
        print(f"\nDemographics:")
        print(f"  Average age: {report['demographics']['age_statistics']['mean']:.1f} years")
        for gender, percentage in report['demographics']['gender_distribution'].items():
            print(f"  {gender.title()}: {percentage:.1%}")
        
        # Diagnosis statistics
        print(f"\nDiagnosis statistics:")
        print(f"  Unique diagnoses: {report['diagnoses']['unique_diagnoses']}")
        print(f"  Avg diagnoses per patient: {report['complexity']['avg_diagnoses_per_patient']:.1f}")
        print(f"  High complexity rate: {report['complexity']['high_complexity_rate']:.1f}%")
        
        # ICU and outcomes
        print(f"\nICU and outcomes:")
        print(f"  ICU admission rate: {report['icu_outcomes']['icu_rate']:.1f}%")
        if report['icu_outcomes']['avg_icu_los']:
            print(f"  Average ICU LOS: {report['icu_outcomes']['avg_icu_los']:.1f} days")
        print(f"  Mortality rate: {report['icu_outcomes']['mortality_rate']:.1f}%")
        
        # Top diagnoses
        print(f"\nTop diagnoses:")
        for diagnosis, count in list(report['diagnoses']['top_diagnoses'].items())[:5]:
            print(f"  {diagnosis}: {count}")
        
        # Top combinations
        print(f"\nTop diagnosis combinations:")
        for combo, count in list(report['diagnoses']['top_combinations'].items())[:5]:
            print(f"  {combo}: {count}")
        
        # Distribution validation
        validation = report['validation']
        print(f"\nDistribution validation:")
        print(f"  Quality: {validation['distribution_quality']}")
        print(f"  Average deviation: {validation['avg_deviation_pct']}%")
        
        if validation['major_deviations']:
            print(f"  Major deviations (>5%):")
            for deviation in validation['major_deviations'][:3]:
                print(f"    {deviation['combination']}: {deviation['difference']}% off target")
        
        print(f"\nOutput files:")
        print(f"  Final dataset: {FINAL_DATA_DIR / 'healthcare_dataset_multi_diagnoses.csv'}")
        print(f"  Report: {report_path}")
        
        logger.info("Dataset combination completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error combining datasets: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check the logs for more details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)