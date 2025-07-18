"""
response_formatter.py

Response formatter for healthcare data generation API.

This module provides functions to format API responses consistently
and prepare data for different output formats.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class HealthcareResponseFormatter:
    """
    Formats API responses for healthcare data generation.
    """
    
    def format_patient_data(self, df: pd.DataFrame, format_type: str = 'json') -> Any:
        """
        Format patient data for API response.
        
        Args:
            df: DataFrame with patient data
            format_type: Output format ('json', 'csv', 'summary')
            
        Returns:
            Formatted data in requested format
        """
        if format_type == 'csv':
            return self._format_as_csv(df)
        elif format_type == 'summary':
            return self._format_as_summary(df)
        else:  # Default to JSON
            return self._format_as_json(df)
    
    def _format_as_json(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format data as JSON-serializable list of dictionaries."""
        # Clean DataFrame for JSON serialization
        df_clean = df.copy()
        
        # Handle NaN values
        df_clean = df_clean.fillna('null')
        
        # Convert numpy types to Python types
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                continue
            elif df_clean[col].dtype in ['int64', 'int32']:
                df_clean[col] = df_clean[col].astype(int)
            elif df_clean[col].dtype in ['float64', 'float32']:
                df_clean[col] = df_clean[col].round(2)
        
        # Convert to list of dictionaries
        records = df_clean.to_dict('records')
        
        # Post-process each record
        for record in records:
            record = self._clean_record(record) # type: ignore
        
        return records # type: ignore
    
    def _format_as_csv(self, df: pd.DataFrame) -> str:
        """Format data as CSV string."""
        return df.to_csv(index=False)
    
    def _format_as_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Format data as summary statistics."""
        summary = {
            'total_patients': len(df),
            'demographics': self._get_demographic_summary(df),
            'clinical_summary': self._get_clinical_summary(df),
            'outcomes': self._get_outcomes_summary(df)
        }
        
        return summary
    
    def _clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean individual patient record for JSON serialization."""
        cleaned = {}
        
        for key, value in record.items():
            # Handle different value types
            if pd.isna(value) or value == 'null':
                cleaned[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                cleaned[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, np.bool_):
                cleaned[key] = bool(value)
            elif isinstance(value, str):
                # Try to parse JSON strings (like diagnoses)
                if key == 'diagnoses' and value.startswith('['):
                    try:
                        cleaned[key] = json.loads(value)
                    except:
                        cleaned[key] = value
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _get_demographic_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get demographic summary statistics."""
        summary = {}
        
        if 'age' in df.columns:
            summary['age'] = {
                'mean': round(df['age'].mean(), 1),
                'median': round(df['age'].median(), 1),
                'min': int(df['age'].min()),
                'max': int(df['age'].max()),
                'std': round(df['age'].std(), 1)
            }
        
        if 'gender' in df.columns:
            summary['gender_distribution'] = df['gender'].value_counts().to_dict()
        
        if 'age_group' in df.columns:
            summary['age_group_distribution'] = df['age_group'].value_counts().to_dict()
        
        if 'ethnicity' in df.columns:
            summary['ethnicity_distribution'] = df['ethnicity'].value_counts().head(5).to_dict()
        
        return summary
    
    def _get_clinical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get clinical characteristics summary."""
        summary = {}
        
        # Diagnosis summary
        if 'diagnoses' in df.columns:
            all_diagnoses = []
            for diagnoses_str in df['diagnoses']:
                try:
                    if isinstance(diagnoses_str, str) and diagnoses_str.startswith('['):
                        diagnoses = json.loads(diagnoses_str)
                        all_diagnoses.extend(diagnoses)
                    elif isinstance(diagnoses_str, list):
                        all_diagnoses.extend(diagnoses_str)
                except:
                    continue
            
            if all_diagnoses:
                diagnosis_counts = pd.Series(all_diagnoses).value_counts()
                summary['top_diagnoses'] = diagnosis_counts.head(10).to_dict()
                summary['unique_diagnoses'] = len(diagnosis_counts)
        
        if 'num_diagnoses' in df.columns:
            summary['diagnoses_per_patient'] = {
                'mean': round(df['num_diagnoses'].mean(), 1),
                'median': round(df['num_diagnoses'].median(), 1),
                'max': int(df['num_diagnoses'].max())
            }
        
        if 'comorbidity_score' in df.columns:
            summary['comorbidity_score'] = {
                'mean': round(df['comorbidity_score'].mean(), 1),
                'median': round(df['comorbidity_score'].median(), 1)
            }
        
        if 'risk_level' in df.columns:
            summary['risk_level_distribution'] = df['risk_level'].value_counts().to_dict()
        
        if 'high_complexity' in df.columns:
            summary['high_complexity_rate'] = round(df['high_complexity'].mean() * 100, 1)
        
        return summary
    
    def _get_outcomes_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get outcomes and utilization summary."""
        summary = {}
        
        # ICU statistics
        if 'has_icu_stay' in df.columns:
            icu_patients = df[df['has_icu_stay'] == 1]
            summary['icu_statistics'] = {
                'admission_rate': round(df['has_icu_stay'].mean() * 100, 1),
                'total_icu_patients': int(df['has_icu_stay'].sum())
            }
            
            if 'icu_los_days' in df.columns and len(icu_patients) > 0:
                summary['icu_statistics']['average_los_days'] = round(
                    icu_patients['icu_los_days'].mean(), 1
                )
        
        # Hospital LOS
        if 'hospital_los_days' in df.columns:
            summary['hospital_los'] = {
                'mean': round(df['hospital_los_days'].mean(), 1),
                'median': round(df['hospital_los_days'].median(), 1)
            }
        
        # Mortality
        if 'mortality' in df.columns:
            summary['mortality_rate'] = round(df['mortality'].mean() * 100, 1)
        
        # Emergency admissions
        if 'emergency_admission' in df.columns:
            summary['emergency_admission_rate'] = round(df['emergency_admission'].mean() * 100, 1)
        
        # Utilization
        if 'utilization_category' in df.columns:
            summary['utilization_distribution'] = df['utilization_category'].value_counts().to_dict()
        
        return summary
    
    def create_generation_metadata(
        self, 
        original_query: str,
        filters_applied: Dict[str, Any],
        num_requested: int,
        num_generated: int,
        generation_time: float = None # type: ignore
    ) -> Dict[str, Any]: # type: ignore
        """
        Create metadata for data generation response.
        
        Args:
            original_query: Original user query
            filters_applied: Filters that were applied
            num_requested: Number of patients requested
            num_generated: Number of patients actually generated
            generation_time: Time taken for generation in seconds
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'original_query': original_query,
            'num_requested': num_requested,
            'num_generated': num_generated,
            'filters_applied': filters_applied,
            'success_rate': round((num_generated / num_requested) * 100, 1) if num_requested > 0 else 0.0
        }
        
        if generation_time is not None:
            metadata['generation_time_seconds'] = round(generation_time, 3)
            metadata['patients_per_second'] = round(num_generated / generation_time, 1) if generation_time > 0 else 0
        
        return metadata
    
    def create_api_response(
        self,
        success: bool,
        data: Any = None,
        message: str = None,
        metadata: Dict[str, Any] = None,
        errors: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create standardized API response.
        
        Args:
            success: Whether the request was successful
            data: Response data
            message: Response message
            metadata: Generation metadata
            errors: List of error messages
            
        Returns:
            Standardized API response dictionary
        """
        response = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if data is not None:
            response['data'] = data
        
        if message:
            response['message'] = message
        
        if metadata:
            response['metadata'] = metadata
        
        if errors:
            response['errors'] = errors
        
        if not success and not errors:
            response['errors'] = ['Unknown error occurred']
        
        return response
    
    def create_error_response(
        self,
        error_message: str,
        error_code: str = None,
        details: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            error_message: Main error message
            error_code: Error code identifier
            details: Additional error details
            
        Returns:
            Error response dictionary
        """
        response = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'error': {
                'message': error_message
            }
        }
        
        if error_code:
            response['error']['code'] = error_code
        
        if details:
            response['error']['details'] = details
        
        return response
    
    def format_validation_errors(self, validation_errors: List[str]) -> Dict[str, Any]:
        """
        Format validation errors for API response.
        
        Args:
            validation_errors: List of validation error messages
            
        Returns:
            Formatted validation error response
        """
        return self.create_error_response(
            error_message="Validation failed",
            error_code="VALIDATION_ERROR",
            details={
                'validation_errors': validation_errors,
                'error_count': len(validation_errors)
            }
        )
    
    def calculate_response_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics for the response.
        
        Args:
            df: Generated patient DataFrame
            
        Returns:
            Dictionary of response statistics
        """
        stats = {
            'record_count': len(df),
            'column_count': len(df.columns),
            'data_quality': {
                'completeness': self._calculate_completeness(df),
                'uniqueness': self._calculate_uniqueness(df)
            }
        }
        
        return stats
    
    def _calculate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data completeness metrics."""
        completeness = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness[col] = round((non_null_count / len(df)) * 100, 1)
        
        return completeness
    
    def _calculate_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data uniqueness metrics."""
        uniqueness = {}
        
        # Overall uniqueness
        if 'patient_id' in df.columns:
            unique_patients = df['patient_id'].nunique()
            uniqueness['unique_patients'] = unique_patients
            uniqueness['duplicate_rate'] = round(((len(df) - unique_patients) / len(df)) * 100, 1)
        
        # Key field uniqueness
        key_fields = ['patient_id', 'age', 'gender']
        for field in key_fields:
            if field in df.columns:
                unique_count = df[field].nunique()
                uniqueness[f'{field}_unique_values'] = unique_count
        
        return uniqueness
    
    def prepare_download_response(
        self,
        df: pd.DataFrame,
        filename: str,
        format_type: str = 'csv'
    ) -> Dict[str, Any]:
        """
        Prepare response for file download.
        
        Args:
            df: DataFrame to download
            filename: Suggested filename
            format_type: File format ('csv', 'json', 'excel')
            
        Returns:
            Download response with content and headers
        """
        if format_type == 'csv':
            content = df.to_csv(index=False)
            content_type = 'text/csv'
            file_extension = '.csv'
        elif format_type == 'json':
            content = df.to_json(orient='records', indent=2)
            content_type = 'application/json'
            file_extension = '.json'
        elif format_type == 'excel':
            # For Excel, we'd need to return bytes, but for simplicity returning CSV
            content = df.to_csv(index=False)
            content_type = 'text/csv'
            file_extension = '.csv'
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        # Ensure filename has correct extension
        if not filename.endswith(file_extension):
            filename += file_extension
        
        return {
            'content': content,
            'filename': filename,
            'content_type': content_type,
            'size_bytes': len(content.encode('utf-8'))
        }