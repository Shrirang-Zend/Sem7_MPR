"""
synthea_processor.py

Synthea data processor for multi-diagnosis healthcare data generation.

This module processes Synthea-generated synthetic healthcare data to create
a standardized dataset with multi-diagnosis support for ICU patient research.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime, timedelta

from config.settings import (
    SYNTHEA_DATA_DIR, PROCESSED_DATA_DIR, MIN_AGE, MAX_AGE,
    RANDOM_SEED, TARGET_DATASET_SIZE, SYNTHEA_TARGET_SIZE
)
from src.utils.common_utils import (
    calculate_age, categorize_age, calculate_los_days, 
    calculate_comorbidity_score, determine_risk_level,
    determine_utilization_category, clean_dataframe,
    safe_json_dumps, generate_patient_id
)
from src.data_processing.disease_mapper import disease_mapper

logger = logging.getLogger(__name__)

class SyntheaProcessor:
    """
    Processes Synthea synthetic healthcare data into standardized format.
    """
    
    def __init__(self):
        self.data_dir = SYNTHEA_DATA_DIR
        self.random_seed = RANDOM_SEED
        np.random.seed(self.random_seed)
        
        # Data containers
        self.patients_df = None
        self.conditions_df = None
        self.procedures_df = None
        self.medications_df = None
        self.encounters_df = None
        self.observations_df = None
        
    def load_synthea_data(self) -> bool:
        """
        Load all required Synthea CSV files.
        
        Returns:
            True if all files loaded successfully, False otherwise
        """
        required_files = {
            'patients': 'patients.csv',
            'conditions': 'conditions.csv',
            'procedures': 'procedures.csv',
            'medications': 'medications.csv',
            'encounters': 'encounters.csv',
            'observations': 'observations.csv'
        }
        
        logger.info(f"Loading Synthea data from: {self.data_dir}")
        
        try:
            # Load each required file
            for data_type, filename in required_files.items():
                file_path = self.data_dir / filename
                
                if not file_path.exists():
                    logger.error(f"Required file not found: {file_path}")
                    return False
                
                df = pd.read_csv(file_path)
                setattr(self, f"{data_type}_df", df)
                logger.info(f"Loaded {filename}: {len(df)} rows")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Synthea data: {e}")
            return False
    
    def process_patient_demographics(self) -> pd.DataFrame:
        """
        Process patient demographics and basic information.
        
        Returns:
            DataFrame with processed patient demographics
        """
        logger.info("Processing patient demographics")
        
        # Start with patients dataframe
        patients = self.patients_df.copy() # type: ignore
        
        # Calculate age
        patients['age'] = patients['BIRTHDATE'].apply(
            lambda x: calculate_age(x, datetime.now())
        )
        
        # Filter by age (adults only)
        patients = patients[
            (patients['age'] >= MIN_AGE) & 
            (patients['age'] <= MAX_AGE)
        ]
        
        # Create standardized columns
        processed_patients = pd.DataFrame({
            'patient_id': patients['Id'].astype(str),
            'age': patients['age'],
            'gender': patients['GENDER'].str.lower(),
            'ethnicity': patients['ETHNICITY'],
            'birthdate': pd.to_datetime(patients['BIRTHDATE']),
            'deathdate': pd.to_datetime(patients['DEATHDATE']),
            'source': 'synthea'
        })
        
        # Add age group
        processed_patients['age_group'] = processed_patients['age'].apply(categorize_age)
        
        # Determine mortality status
        processed_patients['mortality'] = processed_patients['deathdate'].notna().astype(int)
        
        logger.info(f"Processed {len(processed_patients)} patient demographics")
        return processed_patients
    
    def process_patient_diagnoses(self) -> Dict[str, List[str]]:
        """
        Process patient conditions into standardized diagnoses.
        
        Returns:
            Dictionary mapping patient_id to list of diagnoses
        """
        logger.info("Processing patient diagnoses")
        
        # Use disease mapper to process conditions
        patient_diagnoses = disease_mapper.process_synthea_conditions(self.conditions_df) # type: ignore
        
        # Log statistics
        stats = disease_mapper.get_category_statistics(patient_diagnoses)
        logger.info(f"Diagnosis statistics: {stats['total_patients']} patients, "
                   f"avg {stats['avg_diagnoses_per_patient']:.1f} diagnoses per patient")
        
        return patient_diagnoses
    
    def calculate_healthcare_utilization(self, patient_id: str) -> Dict[str, int]:
        """
        Calculate healthcare utilization metrics for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with utilization metrics
        """
        # Count procedures
        num_procedures = len(self.procedures_df[self.procedures_df['PATIENT'] == patient_id]) # type: ignore
        
        # Count medications
        num_medications = len(self.medications_df[self.medications_df['PATIENT'] == patient_id]) # type: ignore
        
        # Count encounters
        encounters = self.encounters_df[self.encounters_df['PATIENT'] == patient_id] # type: ignore
        num_encounters = len(encounters)
        
        return {
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'num_encounters': num_encounters
        }
    
    def simulate_icu_stays(self, processed_patients: pd.DataFrame, 
                          patient_diagnoses: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Simulate ICU stays based on patient diagnoses and characteristics.
        
        Args:
            processed_patients: DataFrame with patient demographics
            patient_diagnoses: Dictionary mapping patient_id to diagnoses
            
        Returns:
            DataFrame with ICU stay information added
        """
        logger.info("Simulating ICU stays")
        
        # ICU probability based on diagnoses
        icu_probabilities = {
            'SEPSIS': 0.8,
            'CARDIOVASCULAR': 0.6,
            'RESPIRATORY': 0.5,
            'NEUROLOGICAL': 0.7,
            'TRAUMA': 0.4,
            'CANCER': 0.3,
            'RENAL': 0.4,
            'DIABETES': 0.2,
            'HYPERTENSION': 0.1,
            'OTHER': 0.2
        }
        
        icu_data = []
        
        for _, patient in processed_patients.iterrows():
            patient_id = patient['patient_id']
            diagnoses = patient_diagnoses.get(patient_id, ['OTHER'])
            
            # Calculate ICU probability
            max_prob = max([icu_probabilities.get(dx, 0.2) for dx in diagnoses])
            
            # Age adjustment
            age_factor = 1.0
            if patient['age'] >= 80:
                age_factor = 1.3
            elif patient['age'] >= 65:
                age_factor = 1.2
            elif patient['age'] < 40:
                age_factor = 0.8
            
            # Multiple diagnoses increase ICU probability
            complexity_factor = 1.0 + (len(diagnoses) - 1) * 0.1
            
            final_prob = min(max_prob * age_factor * complexity_factor, 0.9)
            
            # Determine ICU stay
            has_icu_stay = np.random.random() < final_prob
            
            if has_icu_stay:
                # Generate ICU length of stay based on diagnoses
                if 'SEPSIS' in diagnoses:
                    icu_los_days = np.random.gamma(2, 3)  # 3-8 days average
                elif 'NEUROLOGICAL' in diagnoses:
                    icu_los_days = np.random.gamma(3, 4)  # 5-12 days average
                elif 'RESPIRATORY' in diagnoses:
                    icu_los_days = np.random.gamma(2.5, 3.5)  # 4-10 days average
                elif 'CARDIOVASCULAR' in diagnoses:
                    icu_los_days = np.random.gamma(1.5, 2)  # 2-5 days average
                else:
                    icu_los_days = np.random.gamma(1.8, 2.5)  # 2-6 days average
                
                icu_los_days = max(1, min(icu_los_days, 30))  # Cap between 1-30 days
            else:
                icu_los_days = 0
            
            icu_data.append({
                'patient_id': patient_id,
                'has_icu_stay': int(has_icu_stay),
                'icu_los_days': round(icu_los_days, 1)
            })
        
        # Merge with processed patients
        icu_df = pd.DataFrame(icu_data)
        result = processed_patients.merge(icu_df, on='patient_id', how='left')
        
        logger.info(f"Simulated ICU stays: {result['has_icu_stay'].sum()} patients with ICU stays")
        return result
    
    def simulate_hospital_stays(self, processed_patients: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate hospital stays and admission types.
        
        Args:
            processed_patients: DataFrame with patient data including ICU stays
            
        Returns:
            DataFrame with hospital stay information added
        """
        logger.info("Simulating hospital stays")
        
        hospital_data = []
        
        for _, patient in processed_patients.iterrows():
            # Hospital LOS is always >= ICU LOS
            icu_los = patient['icu_los_days']
            
            if icu_los > 0:
                # ICU patients have longer hospital stays
                base_hospital_los = icu_los + np.random.gamma(2, 3)
            else:
                # Non-ICU patients
                base_hospital_los = np.random.gamma(1.5, 2.5)
            
            hospital_los_days = max(1, min(base_hospital_los, 60))  # Cap between 1-60 days
            
            # Determine admission type
            admission_types = ['Emergency', 'Urgent', 'Elective']
            if patient['has_icu_stay']:
                # ICU patients more likely emergency
                admission_probs = [0.7, 0.2, 0.1]
            else:
                admission_probs = [0.4, 0.3, 0.3]
            
            admission_type = np.random.choice(admission_types, p=admission_probs)
            emergency_admission = int(admission_type == 'Emergency')
            
            # Insurance type simulation
            insurance_types = ['Medicare', 'Medicaid', 'Private', 'Self Pay']
            age = patient['age']
            if age >= 65:
                insurance_probs = [0.8, 0.1, 0.08, 0.02]
            elif age >= 18:
                insurance_probs = [0.1, 0.3, 0.55, 0.05]
            else:
                insurance_probs = [0.05, 0.4, 0.5, 0.05]
            
            insurance = np.random.choice(insurance_types, p=insurance_probs)
            
            hospital_data.append({
                'patient_id': patient['patient_id'],
                'hospital_los_days': round(hospital_los_days, 1),
                'admission_type': admission_type,
                'emergency_admission': emergency_admission,
                'insurance': insurance
            })
        
        # Merge with processed patients
        hospital_df = pd.DataFrame(hospital_data)
        result = processed_patients.merge(hospital_df, on='patient_id', how='left')
        
        logger.info(f"Simulated hospital stays: avg LOS {result['hospital_los_days'].mean():.1f} days")
        return result
    
    def create_final_dataset(self) -> pd.DataFrame:
        """
        Create the final processed Synthea dataset.
        
        Returns:
            Final processed DataFrame
        """
        logger.info("Creating final Synthea dataset")
        
        # Process patient demographics
        processed_patients = self.process_patient_demographics()
        
        # Process diagnoses
        patient_diagnoses = self.process_patient_diagnoses()
        
        # Filter patients who have diagnoses
        patients_with_diagnoses = [
            pid for pid in processed_patients['patient_id'] 
            if pid in patient_diagnoses
        ]
        processed_patients = processed_patients[
            processed_patients['patient_id'].isin(patients_with_diagnoses)
        ]
        
        # Simulate ICU and hospital stays
        processed_patients = self.simulate_icu_stays(processed_patients, patient_diagnoses)
        processed_patients = self.simulate_hospital_stays(processed_patients)
        
        # Add healthcare utilization metrics
        utilization_data = []
        for patient_id in processed_patients['patient_id']:
            utilization = self.calculate_healthcare_utilization(patient_id)
            utilization['patient_id'] = patient_id
            utilization_data.append(utilization)
        
        utilization_df = pd.DataFrame(utilization_data)
        processed_patients = processed_patients.merge(utilization_df, on='patient_id', how='left')
        
        # Add diagnoses as JSON column
        diagnoses_data = []
        for patient_id in processed_patients['patient_id']:
            diagnoses = patient_diagnoses.get(patient_id, ['OTHER'])
            diagnoses_data.append({
                'patient_id': patient_id,
                'diagnoses': safe_json_dumps(diagnoses),
                'num_diagnoses': len(diagnoses)
            })
        
        diagnoses_df = pd.DataFrame(diagnoses_data)
        processed_patients = processed_patients.merge(diagnoses_df, on='patient_id', how='left')
        
        # Calculate derived metrics
        processed_patients['comorbidity_score'] = processed_patients.apply(
            lambda row: calculate_comorbidity_score(
                json.loads(row['diagnoses']) if row['diagnoses'] else []
            ), axis=1
        )
        
        processed_patients['risk_level'] = processed_patients.apply(
            lambda row: determine_risk_level(
                row['age'],
                json.loads(row['diagnoses']) if row['diagnoses'] else [],
                row['comorbidity_score']
            ), axis=1
        )
        
        processed_patients['utilization_category'] = processed_patients.apply(
            lambda row: determine_utilization_category(
                row['hospital_los_days'],
                row['icu_los_days'],
                row['num_procedures'],
                row['num_medications']
            ), axis=1
        )
        
        # Add complexity flag
        processed_patients['high_complexity'] = (
            (processed_patients['num_diagnoses'] >= 3) |
            (processed_patients['comorbidity_score'] >= 5) |
            (processed_patients['icu_los_days'] >= 7)
        ).astype(int)
        
        # Sample to target size if necessary
        if len(processed_patients) > SYNTHEA_TARGET_SIZE:
            processed_patients = processed_patients.sample(
                n=SYNTHEA_TARGET_SIZE, 
                random_state=self.random_seed
            )
        
        # Clean and validate
        processed_patients = clean_dataframe(processed_patients)
        
        logger.info(f"Final Synthea dataset: {len(processed_patients)} patients")
        return processed_patients
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> Path: # type: ignore
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Optional filename (default: synthea_processed.csv)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = 'synthea_processed.csv'
        
        output_path = PROCESSED_DATA_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed Synthea data to: {output_path}")
        
        return output_path
    
    def generate_processing_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a processing report with statistics.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with processing statistics
        """
        # Basic statistics
        total_patients = len(df)
        avg_age = df['age'].mean()
        gender_dist = df['gender'].value_counts(normalize=True).to_dict()
        
        # Diagnosis statistics
        all_diagnoses = []
        for diagnoses_json in df['diagnoses']:
            diagnoses = json.loads(diagnoses_json) if diagnoses_json else []
            all_diagnoses.extend(diagnoses)
        
        diagnosis_counts = pd.Series(all_diagnoses).value_counts()
        
        # ICU statistics
        icu_patients = df[df['has_icu_stay'] == 1]
        icu_rate = len(icu_patients) / total_patients
        avg_icu_los = icu_patients['icu_los_days'].mean() if len(icu_patients) > 0 else 0
        
        # Mortality statistics
        mortality_rate = df['mortality'].mean()
        
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'source': 'synthea',
            'total_patients': total_patients,
            'demographics': {
                'avg_age': round(avg_age, 1),
                'age_range': [int(df['age'].min()), int(df['age'].max())],
                'gender_distribution': gender_dist
            },
            'diagnoses': {
                'total_unique_diagnoses': len(diagnosis_counts),
                'avg_diagnoses_per_patient': round(df['num_diagnoses'].mean(), 1),
                'top_diagnoses': diagnosis_counts.head(10).to_dict()
            },
            'icu_statistics': {
                'icu_admission_rate': round(icu_rate * 100, 1),
                'avg_icu_los_days': round(avg_icu_los, 1),
                'total_icu_patients': len(icu_patients)
            },
            'outcomes': {
                'mortality_rate': round(mortality_rate * 100, 1),
                'avg_hospital_los': round(df['hospital_los_days'].mean(), 1)
            },
            'complexity': {
                'high_complexity_patients': int(df['high_complexity'].sum()),
                'avg_comorbidity_score': round(df['comorbidity_score'].mean(), 1)
            }
        }
        
        return report

def process_synthea_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to process Synthea data.
    
    Returns:
        Tuple of (processed DataFrame, processing report)
    """
    processor = SyntheaProcessor()
    
    # Load data
    if not processor.load_synthea_data():
        raise RuntimeError("Failed to load Synthea data")
    
    # Process data
    processed_df = processor.create_final_dataset()
    
    # Generate report
    report = processor.generate_processing_report(processed_df)
    
    # Save processed data
    output_path = processor.save_processed_data(processed_df)
    
    logger.info(f"Synthea processing completed. Output saved to: {output_path}")
    return processed_df, report