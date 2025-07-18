"""
data_combiner.py

Data combiner for merging Synthea and MIMIC-III datasets.

This module combines processed Synthea and MIMIC-III data according to clinical
literature-based target distributions to create the final balanced dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from collections import defaultdict, Counter
from datetime import datetime

from config.settings import (
    PROCESSED_DATA_DIR, FINAL_DATA_DIR, TARGET_DATASET_SIZE,
    SYNTHEA_TARGET_SIZE, MIMIC_TARGET_SIZE, RANDOM_SEED
)
from config.target_distribution import (
    TARGET_DISTRIBUTION, CO_OCCURRENCE_RULES, AGE_PREVALENCE_MODIFIERS
)
from src.utils.common_utils import (
    safe_json_loads, safe_json_dumps, sample_with_replacement,
    clean_dataframe, create_summary_stats
)

logger = logging.getLogger(__name__)

class DataCombiner:
    """
    Combines and balances Synthea and MIMIC-III datasets according to target distribution.
    """
    
    def __init__(self):
        self.target_distribution = TARGET_DISTRIBUTION
        self.target_size = TARGET_DATASET_SIZE
        self.random_seed = RANDOM_SEED
        np.random.seed(self.random_seed)
        
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed Synthea and MIMIC data.
        
        Returns:
            Tuple of (synthea_df, mimic_df)
        """
        synthea_path = PROCESSED_DATA_DIR / 'synthea_processed.csv'
        mimic_path = PROCESSED_DATA_DIR / 'mimic_processed.csv'
        
        if not synthea_path.exists():
            raise FileNotFoundError(f"Synthea processed data not found: {synthea_path}")
        if not mimic_path.exists():
            raise FileNotFoundError(f"MIMIC processed data not found: {mimic_path}")
        
        logger.info("Loading processed datasets...")
        synthea_df = pd.read_csv(synthea_path)
        mimic_df = pd.read_csv(mimic_path)
        
        logger.info(f"Loaded Synthea: {len(synthea_df)} patients")
        logger.info(f"Loaded MIMIC: {len(mimic_df)} patients")
        
        return synthea_df, mimic_df
    
    def standardize_schema(self, synthea_df: pd.DataFrame, mimic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ensure both datasets have the same schema.
        
        Args:
            synthea_df: Synthea processed data
            mimic_df: MIMIC processed data
            
        Returns:
            Tuple of standardized DataFrames
        """
        logger.info("Standardizing dataset schemas...")
        
        # Required columns for final dataset
        required_columns = [
            'patient_id', 'age', 'gender', 'ethnicity', 'insurance',
            'admission_type', 'diagnoses', 'num_diagnoses', 'num_procedures',
            'num_medications', 'hospital_los_days', 'icu_los_days', 'has_icu_stay',
            'mortality', 'emergency_admission', 'age_group', 'comorbidity_score',
            'high_complexity', 'source', 'risk_level', 'utilization_category'
        ]
        
        # Ensure all required columns exist in both datasets
        for df_name, df in [('Synthea', synthea_df), ('MIMIC', mimic_df)]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"{df_name} missing columns: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    if col in ['num_procedures', 'num_medications', 'num_diagnoses']:
                        df[col] = 0
                    elif col in ['mortality', 'has_icu_stay', 'emergency_admission', 'high_complexity']:
                        df[col] = 0
                    elif col in ['hospital_los_days', 'icu_los_days', 'comorbidity_score']:
                        df[col] = 0.0
                    else:
                        df[col] = 'Unknown'
        
        # Select only required columns
        synthea_standardized = synthea_df[required_columns].copy()
        mimic_standardized = mimic_df[required_columns].copy()
        
        # Ensure consistent data types
        for df in [synthea_standardized, mimic_standardized]:
            df['patient_id'] = df['patient_id'].astype(str)
            df['age'] = df['age'].astype(int)
            df['diagnoses'] = df['diagnoses'].astype(str)
            
        logger.info("Schema standardization completed")
        return synthea_standardized, mimic_standardized
    
    def analyze_current_distribution(self, combined_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze current diagnosis combination distribution.
        
        Args:
            combined_df: Combined dataset
            
        Returns:
            Dictionary with current distribution percentages
        """
        current_distribution = {}
        total_patients = len(combined_df)
        
        # Count diagnosis combinations
        combination_counts = defaultdict(int)
        
        for _, row in combined_df.iterrows():
            diagnoses = safe_json_loads(row['diagnoses'], [])
            if diagnoses:
                # Sort diagnoses for consistent representation
                combination = tuple(sorted(diagnoses))
                combination_counts[combination] += 1
        
        # Convert to percentages
        for combination, count in combination_counts.items():
            current_distribution[combination] = count / total_patients
        
        return current_distribution
    
    def enhance_synthea_comorbidities(self, synthea_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance Synthea data with realistic co-occurrence patterns.
        
        Args:
            synthea_df: Synthea dataset
            
        Returns:
            Enhanced Synthea dataset with improved comorbidities
        """
        logger.info("Enhancing Synthea comorbidities based on clinical literature...")
        
        enhanced_df = synthea_df.copy()
        enhanced_count = 0
        
        for idx, row in enhanced_df.iterrows():
            current_diagnoses = safe_json_loads(row['diagnoses'], [])
            age = row['age']
            age_group = row['age_group']
            
            # Add co-occurring conditions based on clinical rules
            additional_diagnoses = set()
            
            for primary_dx in current_diagnoses:
                if primary_dx in CO_OCCURRENCE_RULES:
                    co_occurrence_probs = CO_OCCURRENCE_RULES[primary_dx]
                    
                    for secondary_dx, base_prob in co_occurrence_probs.items():
                        if secondary_dx not in current_diagnoses:
                            # Adjust probability based on age
                            age_modifier = AGE_PREVALENCE_MODIFIERS.get(age_group, {}).get(secondary_dx, 1.0)
                            adjusted_prob = base_prob * age_modifier
                            
                            # Add some randomness
                            if np.random.random() < adjusted_prob:
                                additional_diagnoses.add(secondary_dx)
            
            # Update diagnoses if new ones were added
            if additional_diagnoses:
                new_diagnoses = sorted(list(set(current_diagnoses) | additional_diagnoses))
                enhanced_df.at[idx, 'diagnoses'] = safe_json_dumps(new_diagnoses)
                enhanced_df.at[idx, 'num_diagnoses'] = len(new_diagnoses)
                enhanced_count += 1
        
        logger.info(f"Enhanced {enhanced_count} patients with additional comorbidities")
        return enhanced_df
    
    def balance_to_target_distribution(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the combined dataset to match target distribution.
        
        Args:
            combined_df: Combined dataset
            
        Returns:
            Balanced dataset matching target distribution
        """
        logger.info("Balancing dataset to target distribution...")
        
        # Group patients by diagnosis combinations
        combination_groups = defaultdict(list)
        
        for idx, row in combined_df.iterrows():
            diagnoses = safe_json_loads(row['diagnoses'], [])
            if diagnoses:
                combination = tuple(sorted(diagnoses))
                combination_groups[combination].append(idx)
            else:
                combination_groups[('OTHER',)].append(idx)
        
        # Create balanced dataset
        balanced_indices = []
        
        for target_combination, target_percentage in self.target_distribution.items():
            target_count = int(self.target_size * target_percentage)
            
            # Find matching combinations (exact match or subset)
            matching_indices = []
            
            # First try exact match
            if target_combination in combination_groups:
                matching_indices.extend(combination_groups[target_combination])
            
            # If not enough, try subset matches
            if len(matching_indices) < target_count:
                for combo, indices in combination_groups.items():
                    # Check if target is a subset of current combination
                    if set(target_combination).issubset(set(combo)) and combo != target_combination:
                        matching_indices.extend(indices)
            
            # Sample or upsample to reach target count
            if len(matching_indices) >= target_count:
                # Downsample
                selected_indices = np.random.choice(
                    matching_indices, size=target_count, replace=False
                )
            elif len(matching_indices) > 0:
                # Upsample with replacement
                selected_indices = np.random.choice(
                    matching_indices, size=target_count, replace=True
                )
            else:
                # No matching patterns found, skip this combination
                logger.warning(f"No patients found for combination: {target_combination}")
                continue
            
            balanced_indices.extend(selected_indices)
        
        # Create balanced dataset
        balanced_df = combined_df.iloc[balanced_indices].copy()
        
        # Remove duplicates that may have resulted from upsampling
        balanced_df = balanced_df.drop_duplicates(subset=['patient_id'])
        
        # If we're still short, fill with random patients
        if len(balanced_df) < self.target_size:
            remaining_count = self.target_size - len(balanced_df)
            remaining_indices = list(set(combined_df.index) - set(balanced_df.index))
            
            if remaining_indices:
                additional_indices = np.random.choice(
                    remaining_indices, 
                    size=min(remaining_count, len(remaining_indices)), 
                    replace=False
                )
                additional_df = combined_df.iloc[additional_indices]
                balanced_df = pd.concat([balanced_df, additional_df], ignore_index=True)
        
        # If we have too many, sample down
        if len(balanced_df) > self.target_size:
            balanced_df = balanced_df.sample(n=self.target_size, random_state=self.random_seed)
        
        logger.info(f"Balanced dataset created: {len(balanced_df)} patients")
        return balanced_df
    
    def recalculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate derived metrics after balancing.
        
        Args:
            df: Dataset to recalculate
            
        Returns:
            Dataset with updated derived metrics
        """
        logger.info("Recalculating derived metrics...")
        
        updated_df = df.copy()
        
        # Update metrics that depend on diagnoses
        for idx, row in updated_df.iterrows():
            diagnoses = safe_json_loads(row['diagnoses'], [])
            
            # Update comorbidity score
            from src.utils.common_utils import calculate_comorbidity_score, determine_risk_level, determine_utilization_category
            
            updated_df.at[idx, 'comorbidity_score'] = calculate_comorbidity_score(diagnoses)
            updated_df.at[idx, 'num_diagnoses'] = len(diagnoses)
            
            # Update risk level
            updated_df.at[idx, 'risk_level'] = determine_risk_level(
                row['age'], diagnoses, updated_df.at[idx, 'comorbidity_score']
            )
            
            # Update utilization category
            updated_df.at[idx, 'utilization_category'] = determine_utilization_category(
                row['hospital_los_days'], row['icu_los_days'],
                row['num_procedures'], row['num_medications']
            )
            
            # Update complexity flag
            updated_df.at[idx, 'high_complexity'] = int(
                len(diagnoses) >= 3 or 
                updated_df.at[idx, 'comorbidity_score'] >= 5 or
                row['icu_los_days'] >= 7
            )
        
        return updated_df
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """
        Create the final combined and balanced dataset.
        
        Returns:
            Final combined dataset
        """
        logger.info("Creating combined dataset...")
        
        # Load processed data
        synthea_df, mimic_df = self.load_processed_data()
        
        # Standardize schemas
        synthea_df, mimic_df = self.standardize_schema(synthea_df, mimic_df)
        
        # Enhance Synthea with better comorbidities
        synthea_df = self.enhance_synthea_comorbidities(synthea_df)
        
        # Combine datasets
        combined_df = pd.concat([synthea_df, mimic_df], ignore_index=True)
        
        # Regenerate patient IDs to avoid conflicts
        combined_df['patient_id'] = [f"PT{i:06d}" for i in range(len(combined_df))]
        
        # Balance to target distribution
        balanced_df = self.balance_to_target_distribution(combined_df)
        
        # Recalculate derived metrics
        final_df = self.recalculate_derived_metrics(balanced_df)
        
        # Clean the dataset
        final_df = clean_dataframe(final_df)
        
        logger.info(f"Final combined dataset: {len(final_df)} patients")
        return final_df
    
    def validate_distribution(self, df: pd.DataFrame) -> Dict[str, any]: # type: ignore
        """
        Validate the final distribution against targets.
        
        Args:
            df: Final dataset
            
        Returns:
            Validation results
        """
        logger.info("Validating final distribution...")
        
        current_dist = self.analyze_current_distribution(df)
        
        validation_results = {
            'total_patients': len(df),
            'target_vs_actual': {},
            'distribution_quality': 'good',
            'major_deviations': []
        }
        
        # Compare with target distribution
        for target_combo, target_pct in self.target_distribution.items():
            actual_pct = current_dist.get(target_combo, 0.0)
            difference = abs(actual_pct - target_pct)
            
            validation_results['target_vs_actual'][str(target_combo)] = {
                'target': round(target_pct * 100, 1),
                'actual': round(actual_pct * 100, 1),
                'difference': round(difference * 100, 1)
            }
            
            # Flag major deviations (>5 percentage points)
            if difference > 0.05:
                validation_results['major_deviations'].append({
                    'combination': str(target_combo),
                    'difference': round(difference * 100, 1)
                })
        
        # Overall quality assessment
        avg_deviation = np.mean([
            abs(current_dist.get(combo, 0) - target_pct)
            for combo, target_pct in self.target_distribution.items()
        ])
        
        if avg_deviation > 0.05:
            validation_results['distribution_quality'] = 'needs_improvement'
        elif avg_deviation > 0.03:
            validation_results['distribution_quality'] = 'fair'
        
        validation_results['avg_deviation_pct'] = round(avg_deviation * 100, 1)
        
        return validation_results
    
    def save_final_dataset(self, df: pd.DataFrame, filename: str = None) -> Path: # type: ignore
        """
        Save the final combined dataset.
        
        Args:
            df: Final dataset
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = 'healthcare_dataset_multi_diagnoses.csv'
        
        output_path = FINAL_DATA_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved final dataset to: {output_path}")
        
        return output_path
    
    def generate_combination_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive report on the final dataset.
        
        Args:
            df: Final dataset
            
        Returns:
            Comprehensive report dictionary
        """
        # Basic statistics
        total_patients = len(df)
        
        # Source distribution
        source_dist = df['source'].value_counts(normalize=True).to_dict()
        
        # Age and gender statistics
        age_stats = df['age'].describe().to_dict()
        gender_dist = df['gender'].value_counts(normalize=True).to_dict()
        
        # Diagnosis statistics
        all_diagnoses = []
        combination_counts = Counter()
        
        for diagnoses_json in df['diagnoses']:
            diagnoses = safe_json_loads(diagnoses_json, [])
            all_diagnoses.extend(diagnoses)
            if diagnoses:
                combination = tuple(sorted(diagnoses))
                combination_counts[combination] += 1
        
        diagnosis_counts = Counter(all_diagnoses)
        
        # ICU and outcomes
        icu_stats = {
            'icu_rate': df['has_icu_stay'].mean() * 100,
            'avg_icu_los': df[df['has_icu_stay'] == 1]['icu_los_days'].mean(),
            'mortality_rate': df['mortality'].mean() * 100
        }
        
        # Complexity statistics
        complexity_stats = {
            'high_complexity_rate': df['high_complexity'].mean() * 100,
            'avg_comorbidity_score': df['comorbidity_score'].mean(),
            'avg_diagnoses_per_patient': df['num_diagnoses'].mean()
        }
        
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'dataset_size': total_patients,
            'source_distribution': source_dist,
            'demographics': {
                'age_statistics': age_stats,
                'gender_distribution': gender_dist
            },
            'diagnoses': {
                'unique_diagnoses': len(diagnosis_counts),
                'top_diagnoses': dict(diagnosis_counts.most_common(10)),
                'top_combinations': {
                    str(combo): count for combo, count in combination_counts.most_common(10)
                }
            },
            'icu_outcomes': icu_stats,
            'complexity': complexity_stats,
            'validation': self.validate_distribution(df)
        }
        
        return report

def combine_datasets() -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to combine Synthea and MIMIC datasets.
    
    Returns:
        Tuple of (final dataset, combination report)
    """
    combiner = DataCombiner()
    
    # Create combined dataset
    final_df = combiner.create_combined_dataset()
    
    # Generate report
    report = combiner.generate_combination_report(final_df)
    
    # Save dataset
    output_path = combiner.save_final_dataset(final_df)
    
    logger.info(f"Dataset combination completed. Output saved to: {output_path}")
    return final_df, report