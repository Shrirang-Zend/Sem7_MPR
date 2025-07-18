"""
data_generator.py

Data generator for synthetic healthcare data.

This module provides functions to generate synthetic healthcare data
using trained CTGAN models with filtering and post-processing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json

from src.utils.common_utils import safe_json_dumps, safe_json_loads

logger = logging.getLogger(__name__)

class HealthcareDataGenerator:
    """
    Generates synthetic healthcare data using CTGAN models.
    """
    
    def __init__(self, ctgan_model=None, original_dataset=None):
        self.ctgan_model = ctgan_model
        self.original_dataset = original_dataset
        
    def generate_patients(self, num_patients: int, filters: Dict[str, Any] = None) -> pd.DataFrame: # type: ignore
        """
        Generate synthetic patients with optional filtering.
        
        Args:
            num_patients: Number of patients to generate
            filters: Optional filters to apply
            
        Returns:
            DataFrame with generated patients
        """
        if self.ctgan_model is None:
            raise ValueError("CTGAN model not loaded")
        
        logger.info(f"Generating {num_patients} synthetic patients")
        
        # Generate initial batch (potentially more than needed for filtering)
        generation_multiplier = 2 if filters and self._has_restrictive_filters(filters) else 1
        initial_count = num_patients * generation_multiplier
        
        synthetic_df = self.ctgan_model.sample(initial_count)
        
        # Post-process for clinical validity
        synthetic_df = self._post_process_data(synthetic_df)
        
        # Apply filters if provided
        if filters:
            synthetic_df = self._apply_filters(synthetic_df, filters)
            
            # Generate more if we don't have enough after filtering
            attempts = 0
            while len(synthetic_df) < num_patients and attempts < 3:
                additional_needed = (num_patients - len(synthetic_df)) * 2
                additional_df = self.ctgan_model.sample(additional_needed)
                additional_df = self._post_process_data(additional_df)
                additional_filtered = self._apply_filters(additional_df, filters)
                
                synthetic_df = pd.concat([synthetic_df, additional_filtered], ignore_index=True)
                attempts += 1
        
        # Return requested number of patients
        result_df = synthetic_df.head(num_patients).copy()
        
        # Add generation metadata
        result_df['generated_timestamp'] = pd.Timestamp.now()
        result_df['generation_method'] = 'CTGAN'
        
        logger.info(f"Successfully generated {len(result_df)} patients")
        return result_df
    
    def _has_restrictive_filters(self, filters: Dict[str, Any]) -> bool:
        """Check if filters are likely to be restrictive."""
        restrictive_conditions = [
            filters.get('diagnoses') and len(filters['diagnoses']) > 2,
            filters.get('age_range') and (filters['age_range'][1] - filters['age_range'][0]) < 20,
            filters.get('mortality') is True,
            filters.get('icu_required') is True,
            filters.get('complexity') == 'high'
        ]
        return any(restrictive_conditions)
    
    def _post_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process synthetic data for clinical validity.
        
        Args:
            df: Raw synthetic data
            
        Returns:
            Clinically valid synthetic data
        """
        processed_df = df.copy()
        
        # Fix age range
        if 'age' in processed_df.columns:
            processed_df['age'] = processed_df['age'].clip(18, 100).round().astype(int)
        
        # Fix length of stay values
        if 'hospital_los_days' in processed_df.columns:
            processed_df['hospital_los_days'] = processed_df['hospital_los_days'].clip(1, 60).round(1)
        
        if 'icu_los_days' in processed_df.columns:
            processed_df['icu_los_days'] = processed_df['icu_los_days'].clip(0, 30).round(1)
            
            # Ensure hospital LOS >= ICU LOS
            if 'hospital_los_days' in processed_df.columns:
                processed_df['hospital_los_days'] = np.maximum(
                    processed_df['hospital_los_days'], 
                    processed_df['icu_los_days']
                )
        
        # Fix binary columns
        binary_columns = [col for col in processed_df.columns if col.startswith('has_')]
        for col in binary_columns:
            processed_df[col] = (processed_df[col] > 0.5).astype(int)
        
        # Reconstruct diagnoses from binary indicators if needed
        if 'diagnoses' not in processed_df.columns and binary_columns:
            processed_df = self._reconstruct_diagnoses(processed_df, binary_columns)
        
        # Fix categorical columns
        processed_df = self._fix_categorical_columns(processed_df)
        
        # Ensure clinical logic consistency
        processed_df = self._ensure_clinical_consistency(processed_df)
        
        return processed_df
    
    def _reconstruct_diagnoses(self, df: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
        """Reconstruct diagnoses column from binary indicators."""
        diagnoses_list = []
        
        for _, row in df.iterrows():
            patient_diagnoses = []
            for col in binary_columns:
                if row.get(col, 0) == 1:
                    diagnosis_name = col.replace('has_', '').upper()
                    patient_diagnoses.append(diagnosis_name)
            
            if not patient_diagnoses:
                patient_diagnoses = ['OTHER']
            
            diagnoses_list.append(safe_json_dumps(patient_diagnoses))
        
        df['diagnoses'] = diagnoses_list
        df['num_diagnoses'] = [len(safe_json_loads(d, [])) for d in diagnoses_list]
        
        return df
    
    def _fix_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix categorical column values."""
        # Gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].fillna('unknown')
            valid_genders = ['male', 'female', 'unknown']
            df['gender'] = df['gender'].apply(
                lambda x: x if x in valid_genders else 'unknown'
            )
        
        # Risk level
        if 'risk_level' in df.columns:
            valid_risk_levels = ['low', 'medium', 'high', 'critical']
            df['risk_level'] = df['risk_level'].fillna('medium')
            df['risk_level'] = df['risk_level'].apply(
                lambda x: x if x in valid_risk_levels else 'medium'
            )
        
        # Utilization category
        if 'utilization_category' in df.columns:
            valid_utilization = ['low', 'medium', 'high']
            df['utilization_category'] = df['utilization_category'].fillna('medium')
            df['utilization_category'] = df['utilization_category'].apply(
                lambda x: x if x in valid_utilization else 'medium'
            )
        
        # Age group
        if 'age_group' in df.columns and 'age' in df.columns:
            df['age_group'] = df['age'].apply(self._categorize_age)
        
        return df
    
    def _categorize_age(self, age: float) -> str:
        """Categorize age into groups."""
        if age < 18:
            return 'pediatric'
        elif age < 40:
            return 'young_adult'
        elif age < 65:
            return 'middle_aged'
        elif age < 80:
            return 'elderly'
        else:
            return 'very_elderly'
    
    def _ensure_clinical_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure clinical logic consistency."""
        # ICU patients should have ICU LOS > 0
        if 'has_icu_stay' in df.columns and 'icu_los_days' in df.columns:
            df.loc[df['has_icu_stay'] == 1, 'icu_los_days'] = np.maximum(
                df.loc[df['has_icu_stay'] == 1, 'icu_los_days'], 1.0
            )
            df.loc[df['has_icu_stay'] == 0, 'icu_los_days'] = 0.0
        
        # High complexity patients should have higher comorbidity scores
        if 'high_complexity' in df.columns and 'comorbidity_score' in df.columns:
            df.loc[df['high_complexity'] == 1, 'comorbidity_score'] = np.maximum(
                df.loc[df['high_complexity'] == 1, 'comorbidity_score'], 3
            )
        
        # Emergency admissions should be more likely for ICU patients
        if 'emergency_admission' in df.columns and 'has_icu_stay' in df.columns:
            # Make 80% of ICU patients emergency admissions
            icu_patients = df['has_icu_stay'] == 1
            emergency_mask = np.random.random(icu_patients.sum()) < 0.8
            df.loc[icu_patients, 'emergency_admission'] = emergency_mask.astype(int)
        
        return df
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataset."""
        filtered_df = df.copy()
        
        # Apply diagnosis filters
        if filters.get('diagnoses'):
            diagnosis_condition = pd.Series([False] * len(filtered_df))
            for diagnosis in filters['diagnoses']:
                if 'diagnoses' in filtered_df.columns:
                    diagnosis_condition |= filtered_df['diagnoses'].str.contains(diagnosis, na=False)
                else:
                    # Check binary indicator columns
                    binary_col = f'has_{diagnosis.lower()}'
                    if binary_col in filtered_df.columns:
                        diagnosis_condition |= (filtered_df[binary_col] == 1)
            
            filtered_df = filtered_df[diagnosis_condition]
        
        # Apply age range filter
        if filters.get('age_range') and 'age' in filtered_df.columns:
            min_age, max_age = filters['age_range']
            filtered_df = filtered_df[
                (filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)
            ]
        
        # Apply gender filter
        if filters.get('gender') and 'gender' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['gender'] == filters['gender']]
        
        # Apply ICU filter
        if filters.get('icu_required') is not None and 'has_icu_stay' in filtered_df.columns:
            icu_value = 1 if filters['icu_required'] else 0
            filtered_df = filtered_df[filtered_df['has_icu_stay'] == icu_value]
        
        # Apply mortality filter
        if filters.get('mortality') is not None and 'mortality' in filtered_df.columns:
            mortality_value = 1 if filters['mortality'] else 0
            filtered_df = filtered_df[filtered_df['mortality'] == mortality_value]
        
        # Apply risk level filter
        if filters.get('risk_level') and 'risk_level' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['risk_level'] == filters['risk_level']]
        
        # Apply complexity filter
        if filters.get('complexity') and 'high_complexity' in filtered_df.columns:
            if filters['complexity'] == 'high':
                filtered_df = filtered_df[filtered_df['high_complexity'] == 1]
            elif filters['complexity'] == 'low':
                filtered_df = filtered_df[filtered_df['high_complexity'] == 0]
        
        # Apply emergency filter
        if filters.get('emergency') is not None and 'emergency_admission' in filtered_df.columns:
            emergency_value = 1 if filters['emergency'] else 0
            filtered_df = filtered_df[filtered_df['emergency_admission'] == emergency_value]
        
        return filtered_df
    
    def get_similar_patients(self, patient_profile: Dict[str, Any], num_similar: int = 10) -> pd.DataFrame:
        """
        Generate patients similar to a given profile.
        
        Args:
            patient_profile: Dictionary describing desired patient characteristics
            num_similar: Number of similar patients to generate
            
        Returns:
            DataFrame with similar patients
        """
        if self.original_dataset is None:
            # Generate new patients with similar characteristics
            filters = self._profile_to_filters(patient_profile)
            return self.generate_patients(num_similar, filters)
        
        # Find similar patients in original dataset
        similar_patients = self._find_similar_in_original(patient_profile, num_similar)
        
        if len(similar_patients) < num_similar:
            # Generate additional similar patients
            additional_needed = num_similar - len(similar_patients)
            filters = self._profile_to_filters(patient_profile)
            generated_similar = self.generate_patients(additional_needed, filters)
            
            similar_patients = pd.concat([similar_patients, generated_similar], ignore_index=True)
        
        return similar_patients.head(num_similar)
    
    def _profile_to_filters(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Convert patient profile to filters."""
        filters = {}
        
        if 'diagnoses' in profile:
            filters['diagnoses'] = profile['diagnoses']
        
        if 'age' in profile:
            age = profile['age']
            filters['age_range'] = (max(18, age - 10), min(100, age + 10))
        
        if 'gender' in profile:
            filters['gender'] = profile['gender']
        
        if 'icu_stay' in profile:
            filters['icu_required'] = profile['icu_stay']
        
        return filters
    
    def _find_similar_in_original(self, profile: Dict[str, Any], num_similar: int) -> pd.DataFrame:
        """Find similar patients in original dataset."""
        if self.original_dataset is None:
            return pd.DataFrame()
        
        # Simple similarity based on key characteristics
        similar_df = self.original_dataset.copy()
        
        # Filter by diagnoses if specified
        if 'diagnoses' in profile:
            diagnosis_condition = pd.Series([False] * len(similar_df))
            for diagnosis in profile['diagnoses']:
                diagnosis_condition |= similar_df['diagnoses'].str.contains(diagnosis, na=False)
            similar_df = similar_df[diagnosis_condition]
        
        # Filter by age range if specified
        if 'age' in profile:
            age = profile['age']
            similar_df = similar_df[
                (similar_df['age'] >= age - 10) & (similar_df['age'] <= age + 10)
            ]
        
        # Filter by gender if specified
        if 'gender' in profile:
            similar_df = similar_df[similar_df['gender'] == profile['gender']]
        
        return similar_df.head(num_similar)