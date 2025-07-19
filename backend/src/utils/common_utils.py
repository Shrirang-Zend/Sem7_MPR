"""
common_utils.py

Common utility functions used across the healthcare data generation system.

This module provides shared utilities for data manipulation, date handling,
and common operations.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
import re
import logging

logger = logging.getLogger(__name__)

def safe_json_loads(json_str: str, default: List = None) -> List: # type: ignore
    """
    Safely parse JSON string, returning default value on error.
    
    Args:
        json_str: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Parsed JSON or default value
    """
    if default is None:
        default = []
    
    if pd.isna(json_str) or json_str == '':
        return default
    
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {json_str}")
        return default

def safe_json_dumps(obj: Any) -> str:
    """
    Safely convert object to JSON string.
    
    Args:
        obj: Object to convert to JSON
        
    Returns:
        JSON string representation
    """
    try:
        return json.dumps(obj)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize to JSON: {obj}, error: {e}")
        return "[]"

def calculate_age(birth_date: Union[str, datetime], reference_date: Union[str, datetime] = None) -> int: # type: ignore
    """
    Calculate age from birth date.
    
    Args:
        birth_date: Birth date as string or datetime
        reference_date: Reference date for age calculation (default: today)
        
    Returns:
        Age in years
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Convert strings to datetime if needed
    if isinstance(birth_date, str):
        birth_date = pd.to_datetime(birth_date)
    if isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)
    
    # Calculate age
    age = reference_date.year - birth_date.year
    if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    
    return max(0, age)  # Ensure non-negative age

def categorize_age(age: int) -> str:
    """
    Categorize age into groups.
    
    Args:
        age: Age in years
        
    Returns:
        Age category string
    """
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

def calculate_los_days(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> float:
    """
    Calculate length of stay in days.
    
    Args:
        start_date: Start date/time
        end_date: End date/time
        
    Returns:
        Length of stay in days (can be fractional)
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return 0.0
    
    # Convert to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Calculate difference in days
    los = (end_date - start_date).total_seconds() / (24 * 3600)
    return max(0.0, los)  # Ensure non-negative LOS

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and removing special characters.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase and remove special characters except spaces
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def find_keywords_in_text(text: str, keywords: List[str]) -> List[str]:
    """
    Find which keywords appear in the given text.
    
    Args:
        text: Text to search in
        keywords: List of keywords to search for
        
    Returns:
        List of found keywords
    """
    if pd.isna(text) or text == '':
        return []
    
    normalized_text = normalize_text(text)
    found_keywords = []
    
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if normalized_keyword in normalized_text:
            found_keywords.append(keyword)
    
    return found_keywords

def map_to_categories(text: str, category_mapping: Dict[str, List[str]]) -> List[str]:
    """
    Map text to categories based on keyword matching.
    
    Args:
        text: Text to categorize
        category_mapping: Dict mapping category names to keyword lists
        
    Returns:
        List of matching categories
    """
    if pd.isna(text) or text == '':
        return []
    
    categories = []
    for category, keywords in category_mapping.items():
        if find_keywords_in_text(text, keywords):
            categories.append(category)
    
    return categories if categories else ['OTHER']

def calculate_comorbidity_score(diagnoses: List[str]) -> int:
    """
    Calculate comorbidity score based on number and type of diagnoses.
    
    Args:
        diagnoses: List of diagnosis categories
        
    Returns:
        Comorbidity score
    """
    if not diagnoses:
        return 0
    
    # Base score is number of diagnoses
    base_score = len(diagnoses)
    
    # Add weights for severe conditions
    severe_conditions = ['SEPSIS', 'CANCER', 'NEUROLOGICAL', 'TRAUMA']
    severity_bonus = sum(1 for dx in diagnoses if dx in severe_conditions)
    
    return min(base_score + severity_bonus, 10)  # Cap at 10

def determine_risk_level(age: int, diagnoses: List[str], comorbidity_score: int) -> str:
    """
    Determine risk level based on patient characteristics.
    
    Args:
        age: Patient age
        diagnoses: List of diagnoses
        comorbidity_score: Comorbidity score
        
    Returns:
        Risk level: 'low', 'medium', 'high', or 'critical'
    """
    # Base risk factors
    risk_score = 0
    
    # Age contribution
    if age >= 80:
        risk_score += 3
    elif age >= 65:
        risk_score += 2
    elif age >= 50:
        risk_score += 1
    
    # Diagnosis contribution
    high_risk_conditions = ['SEPSIS', 'CANCER', 'NEUROLOGICAL', 'TRAUMA']
    medium_risk_conditions = ['CARDIOVASCULAR', 'RESPIRATORY', 'RENAL']
    
    # Count high-risk conditions for young patients
    high_risk_count = sum(1 for dx in diagnoses if dx in high_risk_conditions)
    
    for dx in diagnoses:
        if dx in high_risk_conditions:
            risk_score += 3
        elif dx in medium_risk_conditions:
            risk_score += 2
        else:
            risk_score += 1
    
    # Special boost for young patients with severe conditions
    if age < 40 and high_risk_count >= 2:
        risk_score += 2  # Boost young patients with multiple severe conditions
    elif age < 40 and high_risk_count >= 1 and len(diagnoses) >= 3:
        risk_score += 1  # Modest boost for young patients with 1+ severe + multiple conditions
    
    # Comorbidity contribution
    if comorbidity_score >= 7:
        risk_score += 3
    elif comorbidity_score >= 5:
        risk_score += 2
    elif comorbidity_score >= 3:
        risk_score += 1
    
    # Determine risk level
    if risk_score >= 10:
        return 'critical'
    elif risk_score >= 7:
        return 'high'
    elif risk_score >= 4:
        return 'medium'
    else:
        return 'low'

def determine_utilization_category(hospital_los: float, icu_los: float, 
                                 num_procedures: int, num_medications: int) -> str:
    """
    Determine healthcare utilization category.
    
    Args:
        hospital_los: Hospital length of stay in days
        icu_los: ICU length of stay in days
        num_procedures: Number of procedures
        num_medications: Number of medications
        
    Returns:
        Utilization category: 'low', 'medium', or 'high'
    """
    utilization_score = 0
    
    # Length of stay contribution
    if hospital_los >= 14:
        utilization_score += 3
    elif hospital_los >= 7:
        utilization_score += 2
    elif hospital_los >= 3:
        utilization_score += 1
    
    if icu_los >= 7:
        utilization_score += 3
    elif icu_los >= 3:
        utilization_score += 2
    elif icu_los >= 1:
        utilization_score += 1
    
    # Procedures contribution
    if num_procedures >= 10:
        utilization_score += 2
    elif num_procedures >= 5:
        utilization_score += 1
    
    # Medications contribution
    if num_medications >= 15:
        utilization_score += 2
    elif num_medications >= 8:
        utilization_score += 1
    
    # Determine category
    if utilization_score >= 8:
        return 'high'
    elif utilization_score >= 4:
        return 'medium'
    else:
        return 'low'

def generate_patient_id(prefix: str = 'PT', length: int = 8) -> str:
    """
    Generate a random patient ID.
    
    Args:
        prefix: Prefix for the ID
        length: Length of the numeric part
        
    Returns:
        Generated patient ID
    """
    import random
    numeric_part = ''.join([str(random.randint(0, 9)) for _ in range(length)])
    return f"{prefix}{numeric_part}"

def clean_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame: # type: ignore
    """
    Clean dataframe by removing duplicates and handling missing values.
    
    Args:
        df: DataFrame to clean
        required_columns: List of required columns
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Log basic stats
    logger.info(f"Cleaned DataFrame: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    
    return df_clean

def validate_date_range(df: pd.DataFrame, date_column: str, 
                       min_date: str = None, max_date: str = None) -> pd.DataFrame: # type: ignore
    """
    Validate and filter DataFrame based on date range.
    
    Args:
        df: DataFrame to validate
        date_column: Name of the date column
        min_date: Minimum allowed date (ISO format string)
        max_date: Maximum allowed date (ISO format string)
        
    Returns:
        Filtered DataFrame
    """
    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found in DataFrame")
        return df
    
    # Convert to datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Remove rows with invalid dates
    initial_rows = len(df)
    df = df.dropna(subset=[date_column])
    removed_invalid = initial_rows - len(df)
    
    if removed_invalid > 0:
        logger.info(f"Removed {removed_invalid} rows with invalid dates")
    
    # Apply date range filters
    if min_date:
        min_date = pd.to_datetime(min_date) # type: ignore
        before_filter = len(df)
        df = df[df[date_column] >= min_date]
        logger.info(f"Filtered {before_filter - len(df)} rows before {min_date}")
    
    if max_date:
        max_date = pd.to_datetime(max_date) # type: ignore
        before_filter = len(df)
        df = df[df[date_column] <= max_date]
        logger.info(f"Filtered {before_filter - len(df)} rows after {max_date}")
    
    return df

def sample_with_replacement(df: pd.DataFrame, target_size: int, 
                          random_state: int = 42) -> pd.DataFrame:
    """
    Sample DataFrame with replacement to reach target size.
    
    Args:
        df: DataFrame to sample
        target_size: Target number of rows
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    if len(df) == 0:
        logger.warning("Cannot sample from empty DataFrame")
        return df
    
    if len(df) >= target_size:
        # Sample without replacement
        return df.sample(n=target_size, random_state=random_state)
    else:
        # Sample with replacement
        logger.info(f"Upsampling from {len(df)} to {target_size} rows")
        return df.sample(n=target_size, replace=True, random_state=random_state)

def create_summary_stats(df: pd.DataFrame, group_by: str = None) -> Dict[str, Any]: # type: ignore
    """
    Create summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to summarize
        group_by: Column to group by (optional)
        
    Returns:
        Dictionary of summary statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    # Numeric columns stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns stats
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        stats['categorical_summary'] = {}
        for col in cat_cols[:5]:  # Limit to first 5 categorical columns
            stats['categorical_summary'][col] = df[col].value_counts().head().to_dict()
    
    # Missing values
    missing_values = df.isnull().sum()
    stats['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    # Group by analysis
    if group_by and group_by in df.columns:
        stats['group_analysis'] = df.groupby(group_by).size().to_dict()
    
    return stats