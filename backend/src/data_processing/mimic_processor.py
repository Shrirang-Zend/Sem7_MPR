"""
mimic_processor.py

MIMIC-III data processor for multi-diagnosis healthcare data generation.

This module processes MIMIC-III clinical database to create a standardized
dataset with multi-diagnosis support for ICU patient research.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime, timedelta

from config.settings import (
    MIMIC_BASE_DIR, PROCESSED_DATA_DIR, MIN_AGE, MAX_AGE,
    RANDOM_SEED, MIMIC_TARGET_SIZE
)
from src.utils.common_utils import (
    calculate_age, categorize_age, calculate_los_days,
    calculate_comorbidity_score, determine_risk_level,
    determine_utilization_category, clean_dataframe,
    safe_json_dumps, validate_date_range
)
from src.data_processing.disease_mapper import disease_mapper

logger = logging.getLogger(__name__)

class MimicProcessor:
    """
    Processes MIMIC-III clinical database into standardized format.
    """
    
    def __init__(self):
        self.data_dir = MIMIC_BASE_DIR
        self.random_seed = RANDOM_SEED
        np.random.seed(self.random_seed)
        
        # Data containers
        self.patients_df = None
        self.admissions_df = None
        self.diagnoses_df = None
        self.procedures_df = None
        self.prescriptions_df = None
        self.icustays_df = None
        
    def load_mimic_data(self) -> bool:
        """
        Load all required MIMIC-III CSV files.
        
        Returns:
            True if all files loaded successfully, False otherwise
        """
        required_files = {
            'patients': 'PATIENTS.csv',
            'admissions': 'ADMISSIONS.csv',
            'diagnoses': 'DIAGNOSES_ICD.csv',
            'procedures': 'PROCEDURES_ICD.csv',
            'prescriptions': 'PRESCRIPTIONS.csv',
            'icustays': 'ICUSTAYS.csv'
        }
        
        logger.info(f"Loading MIMIC-III data from: {self.data_dir}")
        
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
            logger.error(f"Error loading MIMIC-III data: {e}")
            return False
    
    def process_patient_demographics(self) -> pd.DataFrame:
        """
        Process patient demographics from MIMIC-III data.
        
        Returns:
            DataFrame with processed patient demographics
        """
        logger.info("Processing MIMIC patient demographics")
        
        # Start with patients dataframe
        patients = self.patients_df.copy() # type: ignore
        
        # Convert date columns
        patients['DOB'] = pd.to_datetime(patients['dob'])
        patients['DOD'] = pd.to_datetime(patients['dod'])
        
        # Calculate age at death or current date
        reference_date = datetime.now()
        patients['age'] = patients.apply(
            lambda row: calculate_age(
                row['DOB'], 
                row['DOD'] if pd.notna(row['DOD']) else reference_date
            ), axis=1
        )
        
        # Filter by age (adults only)
        patients = patients[
            (patients['age'] >= MIN_AGE) & 
            (patients['age'] <= MAX_AGE)
        ]
        
        # Create standardized columns
        processed_patients = pd.DataFrame({
            'patient_id': patients['subject_id'].astype(str),
            'age': patients['age'],
            'gender': patients['gender'].str.lower(),
            'birthdate': patients['DOB'],
            'deathdate': patients['DOD'],
            'source': 'mimic'
        })
        
        # Add ethnicity placeholder (not available in MIMIC patients table)
        processed_patients['ethnicity'] = 'Unknown'
        
        # Add age group
        processed_patients['age_group'] = processed_patients['age'].apply(categorize_age)
        
        # Determine mortality status
        processed_patients['mortality'] = processed_patients['deathdate'].notna().astype(int)
        
        logger.info(f"Processed {len(processed_patients)} MIMIC patient demographics")
        return processed_patients
    
    def process_patient_diagnoses(self) -> Dict[str, List[str]]:
        """
        Process patient diagnoses from MIMIC-III ICD codes.
        
        Returns:
            Dictionary mapping subject_id to list of diagnoses
        """
        logger.info("Processing MIMIC patient diagnoses")
        
        # Use disease mapper to process ICD-9 codes
        patient_diagnoses = disease_mapper.process_mimic_diagnoses(self.diagnoses_df) # type: ignore
        
        # Log statistics
        stats = disease_mapper.get_category_statistics(patient_diagnoses)
        logger.info(f"MIMIC diagnosis statistics: {stats['total_patients']} patients, "
                   f"avg {stats['avg_diagnoses_per_patient']:.1f} diagnoses per patient")
        
        return patient_diagnoses
    
    def process_admissions_data(self, processed_patients: pd.DataFrame) -> pd.DataFrame:
        """
        Process admission data to get hospital stay information.
        
        Args:
            processed_patients: DataFrame with patient demographics
            
        Returns:
            DataFrame with admission information added
        """
        logger.info("Processing MIMIC admissions data")
        
        # Convert date columns
        admissions = self.admissions_df.copy() # type: ignore
        admissions['ADMITTIME'] = pd.to_datetime(admissions['admittime'])
        admissions['DISCHTIME'] = pd.to_datetime(admissions['dischtime'])
        
        # Calculate length of stay
        admissions['hospital_los_days'] = admissions.apply(
            lambda row: calculate_los_days(row['ADMITTIME'], row['DISCHTIME']), axis=1
        )
        
        # Get latest admission for each patient
        latest_admissions = admissions.loc[
            admissions.groupby('subject_id')['ADMITTIME'].idxmax()
        ]
        
        # Create admission data
        admission_data = []
        for _, patient in processed_patients.iterrows():
            subject_id = int(patient['patient_id'])
            
            # Find latest admission
            patient_admission = latest_admissions[
                latest_admissions['subject_id'] == subject_id
            ]
            
            if len(patient_admission) > 0:
                admission = patient_admission.iloc[0]
                
                # Map admission type
                admission_type_mapping = {
                    'EMERGENCY': 'Emergency',
                    'URGENT': 'Urgent', 
                    'ELECTIVE': 'Elective',
                    'NEWBORN': 'Emergency'  # Treat newborn as emergency
                }
                
                admission_type = admission_type_mapping.get(
                    admission['admission_type'], 'Emergency'
                )
                emergency_admission = int(admission_type == 'Emergency')
                
                # Map insurance
                insurance_mapping = {
                    'Medicare': 'Medicare',
                    'Medicaid': 'Medicaid',
                    'Private': 'Private',
                    'Government': 'Government',
                    'Self Pay': 'Self Pay'
                }
                
                insurance = insurance_mapping.get(
                    admission['insurance'], 'Private'
                )
                
                # Get ethnicity from admissions table
                ethnicity = admission.get('ethnicity', 'Unknown')
                
                admission_data.append({
                    'patient_id': patient['patient_id'],
                    'hospital_los_days': round(admission['hospital_los_days'], 1),
                    'admission_type': admission_type,
                    'emergency_admission': emergency_admission,
                    'insurance': insurance,
                    'ethnicity': ethnicity
                })
            else:
                # Default values for patients without admission data
                admission_data.append({
                    'patient_id': patient['patient_id'],
                    'hospital_los_days': 3.0,  # Default LOS
                    'admission_type': 'Emergency',
                    'emergency_admission': 1,
                    'insurance': 'Medicare',
                    'ethnicity': 'Unknown'
                })
        
        # Merge with processed patients
        admission_df = pd.DataFrame(admission_data)
        result = processed_patients.merge(admission_df, on='patient_id', how='left')
        
        # Update ethnicity column
        result['ethnicity'] = result['ethnicity_y']
        result = result.drop(['ethnicity_x', 'ethnicity_y'], axis=1)
        
        logger.info(f"Processed admissions for {len(result)} patients")
        return result
    
    def process_icu_stays(self, processed_patients: pd.DataFrame) -> pd.DataFrame:
        """
        Process ICU stay data from MIMIC-III.
        
        Args:
            processed_patients: DataFrame with patient data
            
        Returns:
            DataFrame with ICU stay information added
        """
        logger.info("Processing MIMIC ICU stays")
        
        # Convert date columns
        icustays = self.icustays_df.copy() # type: ignore
        icustays['INTIME'] = pd.to_datetime(icustays['intime'])
        icustays['OUTTIME'] = pd.to_datetime(icustays['outtime'])
        
        # Calculate ICU length of stay
        icustays['icu_los_days'] = icustays.apply(
            lambda row: calculate_los_days(row['INTIME'], row['OUTTIME']), axis=1
        )
        
        # Aggregate ICU stays per patient
        icu_summary = icustays.groupby('subject_id').agg({
            'icu_los_days': 'sum',  # Total ICU days
            'icustay_id': 'count'   # Number of ICU stays
        }).reset_index()
        
        icu_summary.columns = ['subject_id', 'icu_los_days', 'num_icu_stays']
        
        # Create ICU data for all patients
        icu_data = []
        for _, patient in processed_patients.iterrows():
            subject_id = int(patient['patient_id'])
            
            # Find ICU data
            patient_icu = icu_summary[icu_summary['subject_id'] == subject_id]
            
            if len(patient_icu) > 0:
                icu_info = patient_icu.iloc[0]
                has_icu_stay = 1
                icu_los_days = round(icu_info['icu_los_days'], 1)
            else:
                has_icu_stay = 0
                icu_los_days = 0.0
            
            icu_data.append({
                'patient_id': patient['patient_id'],
                'has_icu_stay': has_icu_stay,
                'icu_los_days': icu_los_days
            })
        
        # Merge with processed patients
        icu_df = pd.DataFrame(icu_data)
        result = processed_patients.merge(icu_df, on='patient_id', how='left')
        
        icu_patients = result[result['has_icu_stay'] == 1]
        logger.info(f"Processed ICU stays: {len(icu_patients)} patients with ICU stays")
        
        return result
    
    def calculate_healthcare_utilization(self, subject_id: int) -> Dict[str, int]:
        """
        Calculate healthcare utilization metrics for a patient.
        
        Args:
            subject_id: Patient subject ID
            
        Returns:
            Dictionary with utilization metrics
        """
        # Count procedures
        num_procedures = len(self.procedures_df[self.procedures_df['subject_id'] == subject_id]) # type: ignore
        
        # Count medications
        num_medications = len(self.prescriptions_df[self.prescriptions_df['subject_id'] == subject_id]) # type: ignore
        
        # Count admissions
        num_encounters = len(self.admissions_df[self.admissions_df['subject_id'] == subject_id]) # type: ignore
        
        return {
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'num_encounters': num_encounters
        }
    
    def create_final_dataset(self) -> pd.DataFrame:
        """
        Create the final processed MIMIC-III dataset.
        
        Returns:
            Final processed DataFrame
        """
        logger.info("Creating final MIMIC dataset")
        
        # Process patient demographics
        processed_patients = self.process_patient_demographics()
        
        # Process diagnoses
        patient_diagnoses = self.process_patient_diagnoses()
        
        # Filter patients who have diagnoses
        patients_with_diagnoses = [
            pid for pid in processed_patients['patient_id']
            if int(pid) in patient_diagnoses
        ]
        processed_patients = processed_patients[
            processed_patients['patient_id'].isin(patients_with_diagnoses)
        ]
        
        # Process admissions and ICU stays
        processed_patients = self.process_admissions_data(processed_patients)
        processed_patients = self.process_icu_stays(processed_patients)
        
        # Add healthcare utilization metrics
        utilization_data = []
        for patient_id in processed_patients['patient_id']:
            subject_id = int(patient_id)
            utilization = self.calculate_healthcare_utilization(subject_id)
            utilization['patient_id'] = patient_id
            utilization_data.append(utilization)
        
        utilization_df = pd.DataFrame(utilization_data)
        processed_patients = processed_patients.merge(utilization_df, on='patient_id', how='left')
        
        # Add diagnoses as JSON column
        diagnoses_data = []
        for patient_id in processed_patients['patient_id']:
            subject_id = int(patient_id)
            diagnoses = patient_diagnoses.get(subject_id, ['OTHER']) # type: ignore
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
        
        # Clean and validate
        processed_patients = clean_dataframe(processed_patients)
        
        logger.info(f"Final MIMIC dataset: {len(processed_patients)} patients")
        return processed_patients
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> Path: # type: ignore
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Optional filename (default: mimic_processed.csv)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = 'mimic_processed.csv'
        
        output_path = PROCESSED_DATA_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed MIMIC data to: {output_path}")
        
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
            'source': 'mimic',
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

def process_mimic_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to process MIMIC-III data.
    
    Returns:
        Tuple of (processed DataFrame, processing report)
    """
    processor = MimicProcessor()
    
    # Load data
    if not processor.load_mimic_data():
        raise RuntimeError("Failed to load MIMIC-III data")
    
    # Process data
    processed_df = processor.create_final_dataset()
    
    # Generate report
    report = processor.generate_processing_report(processed_df)
    
    # Save processed data
    output_path = processor.save_processed_data(processed_df)
    
    logger.info(f"MIMIC processing completed. Output saved to: {output_path}")
    return processed_df, report