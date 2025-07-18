"""
disease_mapper.py

Disease category mapping utilities for healthcare data processing.

This module provides functions to map various disease codes and descriptions
to standardized disease categories for multi-diagnosis analysis.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Dict, Set, Tuple
import re
import logging

from config.settings import DISEASE_CATEGORIES, ICD9_CATEGORY_MAPPING
from src.utils.common_utils import normalize_text, find_keywords_in_text

logger = logging.getLogger(__name__)

class DiseaseMapper:
    """
    Maps disease codes and descriptions to standardized categories.
    """
    
    def __init__(self):
        self.disease_categories = DISEASE_CATEGORIES
        self.icd9_mapping = ICD9_CATEGORY_MAPPING
        
        # Create reverse mapping for efficient lookup
        self._create_keyword_mapping()
        
    def _create_keyword_mapping(self):
        """Create keyword to category mapping for faster lookup."""
        self.keyword_to_category = {}
        for category, keywords in self.disease_categories.items():
            for keyword in keywords:
                normalized_keyword = normalize_text(keyword)
                if normalized_keyword not in self.keyword_to_category:
                    self.keyword_to_category[normalized_keyword] = []
                self.keyword_to_category[normalized_keyword].append(category)
    
    def map_synthea_condition_to_category(self, condition_description: str) -> List[str]:
        """
        Map Synthea condition description to disease categories.
        
        Args:
            condition_description: Description of the medical condition
            
        Returns:
            List of matching disease categories
        """
        if pd.isna(condition_description) or condition_description == '':
            return ['OTHER']
        
        normalized_desc = normalize_text(condition_description)
        categories = set()
        
        # Check each keyword
        for keyword, keyword_categories in self.keyword_to_category.items():
            if keyword in normalized_desc:
                categories.update(keyword_categories)
        
        # Special mappings for common Synthea conditions
        synthea_mappings = {
            'diabetes mellitus': ['DIABETES'],
            'essential hypertension': ['HYPERTENSION'],
            'chronic kidney disease': ['RENAL'],
            'acute myocardial infarction': ['CARDIOVASCULAR'],
            'congestive heart failure': ['CARDIOVASCULAR'],
            'chronic obstructive pulmonary disease': ['RESPIRATORY'],
            'pneumonia': ['RESPIRATORY'],
            'stroke': ['NEUROLOGICAL'],
            'seizure': ['NEUROLOGICAL'],
            'sepsis': ['SEPSIS'],
            'malignant neoplasm': ['CANCER'],
            'fracture': ['TRAUMA']
        }
        
        for condition, cats in synthea_mappings.items():
            if condition in normalized_desc:
                categories.update(cats)
        
        return list(categories) if categories else ['OTHER']
    
    def map_icd9_to_category(self, icd9_code: str) -> List[str]:
        """
        Map ICD-9 code to disease categories.
        
        Args:
            icd9_code: ICD-9 diagnosis code
            
        Returns:
            List of matching disease categories
        """
        if pd.isna(icd9_code) or icd9_code == '':
            return ['OTHER']
        
        # Clean the ICD-9 code
        clean_code = str(icd9_code).strip().upper()
        
        # Try exact match first
        if clean_code in self.icd9_mapping:
            return [self.icd9_mapping[clean_code]]
        
        # Try prefix matching (3-digit codes)
        code_prefix = clean_code[:3]
        if code_prefix in self.icd9_mapping:
            return [self.icd9_mapping[code_prefix]]
        
        # Try 2-digit prefix for broader categories
        code_prefix_2 = clean_code[:2]
        category_mappings_2digit = {
            '25': 'DIABETES',           # Diabetes mellitus
            '40': 'HYPERTENSION',       # Hypertensive disease
            '41': 'CARDIOVASCULAR',     # Ischemic heart disease
            '42': 'CARDIOVASCULAR',     # Other heart disease
            '43': 'CARDIOVASCULAR',     # Cerebrovascular disease
            '49': 'RESPIRATORY',        # Chronic obstructive pulmonary disease
            '58': 'RENAL',             # Nephritis, nephrotic syndrome
            '03': 'SEPSIS',            # Other bacterial diseases
            '99': 'SEPSIS',            # Septicemia
            '14': 'CANCER',            # Malignant neoplasm lip, oral cavity
            '15': 'CANCER',            # Malignant neoplasm digestive organs
            '16': 'CANCER',            # Malignant neoplasm respiratory
            '17': 'CANCER',            # Malignant neoplasm bone, tissue
            '18': 'CANCER',            # Malignant neoplasm genitourinary
            '19': 'CANCER',            # Malignant neoplasm other sites
            '80': 'TRAUMA',            # Fracture skull
            '81': 'TRAUMA',            # Fracture other bones
            '82': 'TRAUMA',            # Dislocation
            '83': 'TRAUMA',            # Sprains and strains
            '84': 'TRAUMA',            # Intracranial injury
            '85': 'TRAUMA',            # Internal injury thorax
            '86': 'TRAUMA',            # Internal injury abdomen
            '87': 'TRAUMA',            # Internal injury other
        }
        
        if code_prefix_2 in category_mappings_2digit:
            return [category_mappings_2digit[code_prefix_2]]
        
        logger.debug(f"No mapping found for ICD-9 code: {icd9_code}")
        return ['OTHER']
    
    def aggregate_patient_diagnoses(self, patient_conditions: List[str], 
                                  source: str = 'synthea') -> List[str]:
        """
        Aggregate all conditions for a patient into unique disease categories.
        
        Args:
            patient_conditions: List of condition descriptions or ICD codes
            source: Data source ('synthea' or 'mimic')
            
        Returns:
            List of unique disease categories for the patient
        """
        all_categories = set()
        
        for condition in patient_conditions:
            if source.lower() == 'synthea':
                categories = self.map_synthea_condition_to_category(condition)
            else:  # mimic
                categories = self.map_icd9_to_category(condition)
            
            all_categories.update(categories)
        
        # Remove 'OTHER' if there are specific categories
        categories_list = list(all_categories)
        if len(categories_list) > 1 and 'OTHER' in categories_list:
            categories_list.remove('OTHER')
        
        # Sort for consistency
        return sorted(categories_list)
    
    def process_synthea_conditions(self, conditions_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Process Synthea conditions data to extract patient diagnoses.
        
        Args:
            conditions_df: DataFrame with Synthea conditions data
            
        Returns:
            Dictionary mapping patient_id to list of disease categories
        """
        logger.info(f"Processing {len(conditions_df)} Synthea conditions")
        
        patient_diagnoses = {}
        
        # Group by patient
        for patient_id, patient_conditions in conditions_df.groupby('PATIENT'):
            condition_descriptions = patient_conditions['DESCRIPTION'].tolist()
            diagnoses = self.aggregate_patient_diagnoses(condition_descriptions, 'synthea')
            patient_diagnoses[patient_id] = diagnoses
        
        logger.info(f"Processed diagnoses for {len(patient_diagnoses)} patients")
        return patient_diagnoses
    
    def process_mimic_diagnoses(self, diagnoses_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Process MIMIC-III diagnoses data to extract patient diagnoses.
        
        Args:
            diagnoses_df: DataFrame with MIMIC-III diagnoses data
            
        Returns:
            Dictionary mapping subject_id to list of disease categories
        """
        logger.info(f"Processing {len(diagnoses_df)} MIMIC diagnoses")
        
        patient_diagnoses = {}
        
        # Group by patient (SUBJECT_ID)
        for subject_id, patient_diagnoses_df in diagnoses_df.groupby('subject_id'):
            icd9_codes = patient_diagnoses_df['icd9_code'].astype(str).tolist()
            diagnoses = self.aggregate_patient_diagnoses(icd9_codes, 'mimic')
            patient_diagnoses[subject_id] = diagnoses
        
        logger.info(f"Processed diagnoses for {len(patient_diagnoses)} patients")
        return patient_diagnoses
    
    def get_category_statistics(self, patient_diagnoses: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """
        Calculate statistics about disease category distribution.
        
        Args:
            patient_diagnoses: Dictionary mapping patient_id to diagnoses
            
        Returns:
            Statistics about category distribution
        """
        # Count individual categories
        category_counts = {}
        combination_counts = {}
        
        for patient_id, diagnoses in patient_diagnoses.items():
            # Count individual categories
            for diagnosis in diagnoses:
                category_counts[diagnosis] = category_counts.get(diagnosis, 0) + 1
            
            # Count combinations
            if len(diagnoses) > 1:
                combination = tuple(sorted(diagnoses))
                combination_counts[combination] = combination_counts.get(combination, 0) + 1
            elif len(diagnoses) == 1:
                combination = tuple(diagnoses)
                combination_counts[combination] = combination_counts.get(combination, 0) + 1
        
        total_patients = len(patient_diagnoses)
        
        stats = {
            'individual_categories': {
                cat: {'count': count, 'percentage': count/total_patients*100}
                for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            },
            'combinations': {
                str(combo): {'count': count, 'percentage': count/total_patients*100}
                for combo, count in sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)[:20]  # Top 20
            },
            'total_patients': total_patients,
            'avg_diagnoses_per_patient': sum(len(dx) for dx in patient_diagnoses.values()) / total_patients
        }
        
        return stats
    
    def validate_diagnoses_mapping(self, patient_diagnoses: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validate the quality of diagnoses mapping.
        
        Args:
            patient_diagnoses: Dictionary mapping patient_id to diagnoses
            
        Returns:
            Validation results
        """
        total_patients = len(patient_diagnoses)
        patients_with_other_only = sum(1 for dx in patient_diagnoses.values() if dx == ['OTHER'])
        patients_with_no_diagnoses = sum(1 for dx in patient_diagnoses.values() if not dx)
        patients_with_multiple_diagnoses = sum(1 for dx in patient_diagnoses.values() if len(dx) > 1)
        
        validation_results = {
            'total_patients': total_patients,
            'patients_with_other_only': patients_with_other_only,
            'patients_with_no_diagnoses': patients_with_no_diagnoses,
            'patients_with_multiple_diagnoses': patients_with_multiple_diagnoses,
            'percentage_other_only': patients_with_other_only / total_patients * 100 if total_patients > 0 else 0,
            'percentage_no_diagnoses': patients_with_no_diagnoses / total_patients * 100 if total_patients > 0 else 0,
            'percentage_multiple_diagnoses': patients_with_multiple_diagnoses / total_patients * 100 if total_patients > 0 else 0,
            'mapping_quality': 'good' if patients_with_other_only / total_patients < 0.2 else 'needs_improvement'
        }
        
        return validation_results

# Create global mapper instance
disease_mapper = DiseaseMapper()