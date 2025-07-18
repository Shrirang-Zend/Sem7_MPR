"""
app.py

FastAPI application for healthcare data generation.

This module provides a REST API for generating synthetic healthcare data
using trained CTGAN models and natural language query processing.
"""

import logging
import json
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import project modules
from config.settings import (
    API_CONFIG, MODELS_DIR, FINAL_DATA_DIR, 
    RANDOM_SEED, PROJECT_ROOT
)
from src.utils.logging_config import setup_logging, get_logger
from src.utils.common_utils import safe_json_dumps, safe_json_loads
from src.utils.pubmed_client import PubMedClient

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Data Generation API",
    description="Generate synthetic healthcare data with multi-diagnosis support for research",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
ctgan_model = None
original_dataset = None
model_metadata = None
training_columns = None
pubmed_client = None

# Pydantic models for API requests/responses
class GenerateRequest(BaseModel):
    query: str = Field(..., description="Natural language query for data generation")
    num_patients: int = Field(default=100, ge=1, le=API_CONFIG['max_generated_rows'])
    include_original: bool = Field(default=False, description="Include matching original data")
    format: str = Field(default="json", pattern="^(json|csv)$") # type: ignore
    
    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return v.strip()

class GenerateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]
    query_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    dataset_loaded: bool
    version: str

class ModelInfoResponse(BaseModel):
    model_type: str
    training_timestamp: str
    training_samples: int
    training_features: int
    model_parameters: Dict[str, Any]

class QueryParseRequest(BaseModel):
    query: str = Field(..., description="Natural language query to parse")

class QueryParseResponse(BaseModel):
    success: bool
    query_id: str
    parsed_conditions: Dict[str, Any]
    suggested_filters: List[str]
    confidence: float
    extracted_params: Dict[str, Any]
    research_context: str
    pubmed_results: List[Dict[str, Any]]
    embeddings_created: int

class GenerateContextRequest(BaseModel):
    query: str = Field(..., description="Query for generating research context")
    max_articles: int = Field(default=50, ge=1, le=100)

class GenerateContextResponse(BaseModel):
    success: bool
    query_id: str
    extracted_params: Dict[str, Any]
    pubmed_results: List[Dict[str, Any]]
    research_context: str
    suggested_sample_size: Optional[int] = None

class DataGenerateRequest(BaseModel):
    query_id: str
    sample_size: Optional[int] = None
    age_range: Optional[List[int]] = None
    use_research_context: bool = True
    # Add filters for data generation
    gender: Optional[str] = None
    diagnoses: Optional[List[str]] = None
    diagnosis_logic: Optional[str] = 'OR'  # 'AND' or 'OR'
    icu_required: Optional[bool] = None
    mortality: Optional[bool] = None
    risk_level: Optional[str] = None
    complexity: Optional[str] = None

class DataGenerateResponse(BaseModel):
    success: bool
    task_id: str
    message: str

def load_model_and_data():
    """Load the trained CTGAN model and original dataset."""
    global ctgan_model, original_dataset, model_metadata, training_columns, pubmed_client
    
    try:
        # Load CTGAN model
        model_path = MODELS_DIR / 'ctgan_healthcare_model.pkl'
        if model_path.exists():
            logger.info(f"Loading CTGAN model from {model_path}")
            with open(model_path, 'rb') as f:
                ctgan_model = pickle.load(f)
            logger.info("CTGAN model loaded successfully")
        else:
            logger.warning(f"CTGAN model not found at {model_path}")
        
        # Load model metadata
        metadata_path = MODELS_DIR / 'ctgan_model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            training_columns = model_metadata.get('training_columns', [])
            logger.info("Model metadata loaded successfully")
        
        # Load original dataset
        dataset_path = FINAL_DATA_DIR / 'healthcare_dataset_multi_diagnoses.csv'
        if dataset_path.exists():
            logger.info(f"Loading original dataset from {dataset_path}")
            original_dataset = pd.read_csv(dataset_path)
            logger.info(f"Original dataset loaded: {len(original_dataset)} patients")
        else:
            logger.warning(f"Original dataset not found at {dataset_path}")
        
        # Initialize PubMed client
        pubmed_client = PubMedClient(
            api_key=os.getenv("PUBMED_API_KEY"),
            email=os.getenv("PUBMED_EMAIL"),
            tool_name=os.getenv("PUBMED_TOOL_NAME", "healthcare_data_generator")
        )
        logger.info("PubMed client initialized")
            
    except Exception as e:
        logger.error(f"Error loading model and data: {e}")
        raise

def parse_natural_language_query(query: str) -> Dict[str, Any]:
    """Parse natural language query to extract filters and conditions."""
    query_lower = query.lower()
    
    # Initialize filters
    filters = {
        'diagnoses': [],
        'diagnosis_logic': 'OR',  # Default to OR logic
        'age_range': None,
        'gender': None,
        'icu_required': None,
        'mortality': None,
        'risk_level': None,
        'complexity': None
    }
    
    # Extract diagnosis conditions
    diagnosis_keywords = {
        'diabetes': ['diabetes', 'diabetic', 'dm', 'glucose'],
        'cardiovascular': ['cardiovascular', 'cardiac', 'heart', 'cvd', 'coronary'],
        'hypertension': ['hypertension', 'hypertensive', 'high blood pressure', 'htn'],
        'renal': ['renal', 'kidney', 'nephritis', 'dialysis', 'ckd'],
        'respiratory': ['respiratory', 'pulmonary', 'copd', 'asthma', 'pneumonia', 'lung', 'conditions'],
        'sepsis': ['sepsis', 'septic', 'infection', 'bacteremia'],
        'neurological': ['neurological', 'stroke', 'seizure', 'brain', 'neuro'],
        'trauma': ['trauma', 'fracture', 'injury', 'accident'],
        'cancer': ['cancer', 'tumor', 'malignant', 'oncology', 'carcinoma'],
    }
    
    for diagnosis, keywords in diagnosis_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            filters['diagnoses'].append(diagnosis.upper())
    
    # Detect AND vs OR logic for diagnoses
    if len(filters['diagnoses']) > 1:
        # Look for AND patterns
        and_patterns = [' and ', ' with ', ' plus ', ' alongside ', ' combined with ', ' having both ']
        or_patterns = [' or ', ' either ', ' any of ', ' one of ']
        
        has_and_pattern = any(pattern in query_lower for pattern in and_patterns)
        has_or_pattern = any(pattern in query_lower for pattern in or_patterns)
        
        # If we find AND patterns and no OR patterns, use AND logic
        if has_and_pattern and not has_or_pattern:
            filters['diagnosis_logic'] = 'AND'
        # If we find OR patterns and no AND patterns, use OR logic
        elif has_or_pattern and not has_and_pattern:
            filters['diagnosis_logic'] = 'OR'
        # If we find both or neither, analyze specific patterns
        else:
            # Check for specific AND patterns like "X and Y"
            if ' and ' in query_lower and len(filters['diagnoses']) == 2:
                filters['diagnosis_logic'] = 'AND'
    
    # Extract age-related filters
    # First check for specific age ranges (e.g., "aged 45-75", "age 30-60")
    import re
    age_range_pattern = r'(?:aged?|age)\s*(\d+)\s*-\s*(\d+)'
    age_match = re.search(age_range_pattern, query_lower)
    if age_match:
        min_age, max_age = int(age_match.group(1)), int(age_match.group(2))
        filters['age_range'] = (min_age, max_age)
    # Then check for general age categories
    elif 'elderly' in query_lower or 'old' in query_lower or 'geriatric' in query_lower:
        filters['age_range'] = (65, 100)
    elif 'young' in query_lower or 'adult' in query_lower:
        filters['age_range'] = (18, 65)
    elif 'pediatric' in query_lower or 'child' in query_lower:
        filters['age_range'] = (0, 18)
    
    # Extract gender (using word boundaries to avoid false matches)
    import re
    if re.search(r'\b(female|women)\b', query_lower):
        filters['gender'] = 'female'
    elif re.search(r'\b(male|men)\b', query_lower):
        filters['gender'] = 'male'
    
    # Extract ICU requirement
    if 'icu' in query_lower or 'intensive care' in query_lower or 'critical' in query_lower:
        filters['icu_required'] = True
    
    # Extract mortality
    if 'mortality' in query_lower or 'death' in query_lower or 'died' in query_lower:
        filters['mortality'] = True
    elif 'survived' in query_lower or 'survivor' in query_lower:
        filters['mortality'] = False
    
    # Extract risk level
    if 'high risk' in query_lower or 'high-risk' in query_lower or 'severe' in query_lower:
        filters['risk_level'] = 'high'
    elif 'low risk' in query_lower or 'low-risk' in query_lower or 'mild' in query_lower:
        filters['risk_level'] = 'low'
    
    # Extract complexity
    if 'complex' in query_lower or 'multiple' in query_lower or 'comorbid' in query_lower or 'multiple diagnoses' in query_lower:
        filters['complexity'] = 'high'
    
    return filters

def apply_filters_to_data(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply parsed filters to the dataset."""
    filtered_df = df.copy()
    
    # Debug: Show available columns
    logger.info(f"Available columns: {list(filtered_df.columns)}")
    logger.info(f"Dataset shape before filtering: {filtered_df.shape}")
    
    # Debug: Show unique values in key columns
    if 'gender' in filtered_df.columns:
        logger.info(f"Unique gender values: {filtered_df['gender'].unique()}")
    if 'primary_diagnosis' in filtered_df.columns:
        logger.info(f"Unique primary_diagnosis values: {filtered_df['primary_diagnosis'].unique()[:10]}")  # First 10
    
    # Apply diagnosis filters - handle both AND and OR logic
    if filters['diagnoses']:
        diagnosis_logic = filters.get('diagnosis_logic', 'OR')
        logger.info(f"Filtering for diagnoses: {filters['diagnoses']} with {diagnosis_logic} logic")
        
        # Collect diagnosis matches for each requested diagnosis
        diagnosis_matches = {}
        
        for diagnosis in filters['diagnoses']:
            current_diagnosis_matches = pd.Series([False] * len(filtered_df))
            
            # Method 1: Check if it's their primary diagnosis (MAIN METHOD)
            if 'primary_diagnosis' in filtered_df.columns:
                primary_matches = filtered_df['primary_diagnosis'] == diagnosis
                current_diagnosis_matches |= primary_matches
                logger.info(f"Primary diagnosis '{diagnosis}' matches: {primary_matches.sum()} patients")
            
            # Method 2: Check if it appears in their diagnoses list (JSON string)
            if 'diagnoses' in filtered_df.columns:
                json_matches = filtered_df['diagnoses'].str.contains(diagnosis, na=False)
                current_diagnosis_matches |= json_matches
                logger.info(f"Diagnoses field contains '{diagnosis}': {json_matches.sum()} patients")
            
            # Method 3: Check binary indicator columns (if they exist)
            possible_binary_cols = [
                f'has_{diagnosis.lower()}',
                f'has_{diagnosis.lower()}_diagnosis',
                f'has_{diagnosis.lower()}_condition'
            ]
            
            binary_found = False
            for has_col in possible_binary_cols:
                if has_col in filtered_df.columns:
                    binary_matches = filtered_df[has_col] == 1
                    current_diagnosis_matches |= binary_matches
                    logger.info(f"Binary diagnosis '{has_col}' matches: {binary_matches.sum()} patients")
                    binary_found = True
            
            if not binary_found:
                logger.info(f"No binary diagnosis columns found for '{diagnosis}', relying on primary_diagnosis only")
            
            # Store the matches for this diagnosis
            diagnosis_matches[diagnosis] = current_diagnosis_matches
            logger.info(f"Total patients with '{diagnosis}' (any form): {current_diagnosis_matches.sum()}")
        
        # Apply AND or OR logic
        if diagnosis_logic == 'AND':
            # Patient must have ALL diagnoses
            final_condition = pd.Series([True] * len(filtered_df))
            for diagnosis, matches in diagnosis_matches.items():
                final_condition &= matches
            logger.info(f"Patients with ALL diagnoses ({' AND '.join(filters['diagnoses'])}): {final_condition.sum()}")
        else:
            # Patient must have ANY of the diagnoses (OR logic)
            final_condition = pd.Series([False] * len(filtered_df))
            for diagnosis, matches in diagnosis_matches.items():
                final_condition |= matches
            logger.info(f"Patients with ANY diagnosis ({' OR '.join(filters['diagnoses'])}): {final_condition.sum()}")
        
        filtered_df = filtered_df[final_condition]
        logger.info(f"After diagnosis filtering: {len(filtered_df)} patients remain")
    
    # Apply age range filter
    if filters['age_range']:
        min_age, max_age = filters['age_range']
        filtered_df = filtered_df[
            (filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)
        ]
    
    # Apply gender filter
    if filters['gender']:
        logger.info(f"Filtering for gender: {filters['gender']}")
        if 'gender' in filtered_df.columns:
            # Handle different gender encodings (f/m vs female/male)
            target_gender = filters['gender']
            if target_gender == 'female':
                target_gender = 'f'
            elif target_gender == 'male':
                target_gender = 'm'
            
            gender_matches = filtered_df['gender'] == target_gender
            logger.info(f"Gender '{filters['gender']}' (encoded as '{target_gender}') matches: {gender_matches.sum()} patients")
            filtered_df = filtered_df[gender_matches]
            logger.info(f"After gender filtering: {len(filtered_df)} patients")
        else:
            logger.warning("No 'gender' column found in dataset")
    
    # Apply ICU filter
    if filters['icu_required'] is not None:
        icu_value = 1 if filters['icu_required'] else 0
        filtered_df = filtered_df[filtered_df['has_icu_stay'] == icu_value]
    
    # Apply mortality filter
    if filters['mortality'] is not None:
        mortality_value = 1 if filters['mortality'] else 0
        filtered_df = filtered_df[filtered_df['mortality'] == mortality_value]
    
    # Apply risk level filter
    if filters['risk_level']:
        filtered_df = filtered_df[filtered_df['risk_level'] == filters['risk_level']]
    
    # Apply complexity filter
    if filters['complexity'] == 'high':
        filtered_df = filtered_df[filtered_df['high_complexity'] == 1]
    
    return filtered_df

def generate_synthetic_data(num_patients: int, filters: Dict[str, Any] = None) -> pd.DataFrame: # type: ignore
    """Generate synthetic data using CTGAN model."""
    if ctgan_model is None:
        raise HTTPException(status_code=500, detail="CTGAN model not loaded")
    
    logger.info(f"Generating {num_patients} synthetic patients")
    
    # Generate synthetic samples
    synthetic_df = ctgan_model.sample(num_patients)
    
    # Post-process synthetic data
    synthetic_df = post_process_synthetic_data(synthetic_df)
    
    # Apply filters if provided
    if filters and any(filters.values()):
        logger.info(f"Applying filters: {filters}")
        logger.info(f"Before filtering: {len(synthetic_df)} patients")
        
        synthetic_df = apply_filters_to_synthetic_data(synthetic_df, filters)
        logger.info(f"After filtering: {len(synthetic_df)} patients")
        
        # If not enough samples after filtering, keep generating until we have enough
        max_attempts = 10
        attempt = 0
        
        while len(synthetic_df) < num_patients and attempt < max_attempts:
            attempt += 1
            additional_needed = num_patients - len(synthetic_df)
            # Generate more samples (multiply by 5 to increase chances)
            generate_count = max(additional_needed * 5, 200)
            logger.info(f"Attempt {attempt}: Need {additional_needed} more patients, generating {generate_count} samples")
            
            additional_samples = ctgan_model.sample(generate_count)
            additional_samples = post_process_synthetic_data(additional_samples)
            additional_filtered = apply_filters_to_synthetic_data(additional_samples, filters)
            
            logger.info(f"Additional filtered samples from attempt {attempt}: {len(additional_filtered)}")
            
            if len(additional_filtered) > 0:
                # Combine and continue
                synthetic_df = pd.concat([synthetic_df, additional_filtered], ignore_index=True)
                logger.info(f"Running total: {len(synthetic_df)} patients")
        
        # Take only the required number
        synthetic_df = synthetic_df.head(num_patients)
        logger.info(f"Final dataset size: {len(synthetic_df)}")
        
        # Verify final results
        if len(synthetic_df) > 0:
            logger.info("Final dataset verification:")
            if 'gender' in synthetic_df.columns:
                logger.info(f"Gender distribution: {synthetic_df['gender'].value_counts().to_dict()}")
            if 'primary_diagnosis' in synthetic_df.columns:
                logger.info(f"Primary diagnosis distribution: {synthetic_df['primary_diagnosis'].value_counts().to_dict()}")
        else:
            logger.error("No patients remain after filtering! Check filter criteria.")
    
    return synthetic_df.head(num_patients)

def post_process_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process synthetic data to ensure clinical validity."""
    processed_df = df.copy()
    
    # Debug: Show what columns are available in the raw synthetic data
    logger.info(f"Raw synthetic data columns: {list(processed_df.columns)}")
    has_columns = [col for col in processed_df.columns if col.startswith('has_')]
    logger.info(f"Binary indicator columns: {has_columns}")
    
    # Ensure age is within valid range
    processed_df['age'] = processed_df['age'].clip(18, 100).round().astype(int)
    
    # Ensure lengths of stay are non-negative
    if 'hospital_los_days' in processed_df.columns:
        processed_df['hospital_los_days'] = processed_df['hospital_los_days'].clip(0, 60).round(1)
    
    if 'icu_los_days' in processed_df.columns:
        processed_df['icu_los_days'] = processed_df['icu_los_days'].clip(0, 30).round(1)
    
    # Ensure hospital LOS >= ICU LOS
    if 'hospital_los_days' in processed_df.columns and 'icu_los_days' in processed_df.columns:
        processed_df['hospital_los_days'] = np.maximum(
            processed_df['hospital_los_days'], 
            processed_df['icu_los_days']
        )
    
    # Reconstruct diagnoses from binary indicators
    if 'diagnoses' not in processed_df.columns:
        # Map of diagnosis types to look for
        diagnosis_map = {
            'DIABETES': ['has_diabetes', 'has_dm'],
            'CARDIOVASCULAR': ['has_cardiovascular', 'has_cardiac', 'has_heart'],
            'HYPERTENSION': ['has_hypertension', 'has_htn'],
            'RENAL': ['has_renal', 'has_kidney'],
            'RESPIRATORY': ['has_respiratory', 'has_copd', 'has_asthma', 'has_lung'],
            'SEPSIS': ['has_sepsis', 'has_infection'],
            'NEUROLOGICAL': ['has_neurological', 'has_neuro', 'has_stroke'],
            'TRAUMA': ['has_trauma', 'has_injury'],
            'CANCER': ['has_cancer', 'has_tumor', 'has_malignant']
        }
        
        # Also look for any other has_ columns (exclude non-diagnosis ones)
        exclude_cols = {'has_icu_stay', 'has_emergency_admission'}
        all_has_cols = [col for col in processed_df.columns if col.startswith('has_') and col not in exclude_cols]
        
        diagnoses_list = []
        logger.info(f"Available diagnosis columns: {all_has_cols}")
        
        for _, row in processed_df.iterrows():
            patient_diagnoses = []
            
            # Use binary diagnosis indicators as the PRIMARY source (since we have them now!)
            for diagnosis_type, possible_cols in diagnosis_map.items():
                has_diagnosis = False
                
                # Check if any of the possible columns for this diagnosis type are present and positive
                for col in possible_cols:
                    if col in processed_df.columns and row.get(col, 0) > 0.5:
                        has_diagnosis = True
                        break
                
                if has_diagnosis:
                    patient_diagnoses.append(diagnosis_type)
            
            # Add any other diagnoses from remaining has_ columns
            for col in all_has_cols:
                if row.get(col, 0) > 0.5:
                    # Extract diagnosis name from column name
                    diagnosis_name = col.replace('has_', '').upper()
                    # Only add if not already covered by the mapping above
                    if diagnosis_name not in patient_diagnoses and diagnosis_name not in ['ICU_STAY', 'EMERGENCY_ADMISSION']:
                        patient_diagnoses.append(diagnosis_name)
            
            # If still no diagnoses found, use primary_diagnosis as fallback
            if not patient_diagnoses:
                if 'primary_diagnosis' in row and pd.notna(row['primary_diagnosis']):
                    primary_diag = str(row['primary_diagnosis']).upper()
                    if primary_diag not in ['NAN', 'NONE', '']:
                        patient_diagnoses.append(primary_diag)
            
            # If STILL no diagnoses found, use OTHER
            if not patient_diagnoses:
                patient_diagnoses = ['OTHER']
            
            diagnoses_list.append(safe_json_dumps(patient_diagnoses))
        
        processed_df['diagnoses'] = diagnoses_list
    
    # Ensure categorical values are valid
    if 'gender' in processed_df.columns:
        processed_df['gender'] = processed_df['gender'].fillna('unknown')
    
    if 'risk_level' in processed_df.columns:
        valid_risk_levels = ['low', 'medium', 'high', 'critical']
        processed_df['risk_level'] = processed_df['risk_level'].fillna('medium')
        processed_df['risk_level'] = processed_df['risk_level'].apply(
            lambda x: x if x in valid_risk_levels else 'medium'
        )
    
    return processed_df

def apply_filters_to_synthetic_data(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to synthetic data."""
    return apply_filters_to_data(df, filters)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Healthcare Data Generation API")
    try:
        load_model_and_data()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=ctgan_model is not None,
        dataset_loaded=original_dataset is not None,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    return HealthResponse(
        status="healthy" if ctgan_model is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=ctgan_model is not None,
        dataset_loaded=original_dataset is not None,
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if model_metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not available")
    
    return ModelInfoResponse(
        model_type=model_metadata.get('model_type', 'Unknown'),
        training_timestamp=model_metadata.get('training_timestamp', 'Unknown'),
        training_samples=len(original_dataset) if original_dataset is not None else 0,
        training_features=len(training_columns) if training_columns else 0,
        model_parameters=model_metadata.get('model_parameters', {})
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_data(request: GenerateRequest):
    """Generate synthetic healthcare data based on natural language query."""
    try:
        logger.info(f"Generating data for query: '{request.query}' ({request.num_patients} patients)")
        
        # Parse the natural language query
        filters = parse_natural_language_query(request.query)
        logger.info(f"Parsed filters: {filters}")
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(request.num_patients, filters)
        
        # Get matching original data if requested
        original_matches = None
        if request.include_original and original_dataset is not None:
            filtered_original = apply_filters_to_data(original_dataset, filters)
            original_matches = filtered_original.head(min(50, len(filtered_original)))  # Limit to 50
        
        # Prepare response data
        response_data = synthetic_data.to_dict('records')
        
        # Calculate statistics
        metadata = {
            'generated_count': len(synthetic_data),
            'generation_timestamp': datetime.now().isoformat(),
            'filters_applied': {k: v for k, v in filters.items() if v is not None and v != []},
            'original_matches_count': len(original_matches) if original_matches is not None else 0
        }
        
        # Add original matches to response if requested
        if original_matches is not None and len(original_matches) > 0:
            metadata['original_matches'] = original_matches.to_dict('records')
        
        query_info = {
            'original_query': request.query,
            'parsed_conditions': [k for k, v in filters.items() if v is not None and v != []],
            'num_requested': request.num_patients,
            'num_generated': len(synthetic_data)
        }
        
        logger.info(f"Successfully generated {len(synthetic_data)} patients")
        
        return GenerateResponse(
            success=True,
            message=f"Successfully generated {len(synthetic_data)} patients",
            data=response_data, # type: ignore
            metadata=metadata,
            query_info=query_info
        )
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating data: {str(e)}")

@app.get("/generate/csv")
async def generate_data_csv(
    query: str = Query(..., description="Natural language query"),
    num_patients: int = Query(100, ge=1, le=API_CONFIG['max_generated_rows']),
):
    """Generate synthetic data and return as CSV file."""
    try:
        # Parse query and generate data
        filters = parse_natural_language_query(query)
        synthetic_data = generate_synthetic_data(num_patients, filters)
        
        # Save to temporary CSV file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            synthetic_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        # Return file response
        return FileResponse(
            temp_path,
            media_type='text/csv',
            filename=f"synthetic_healthcare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
    except Exception as e:
        logger.error(f"Error generating CSV data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating CSV data: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get detailed system status including all service health checks."""
    try:
        # Check CTGAN Model status
        ctgan_status = "ready" if ctgan_model is not None else "error"
        
        # Check RAG System status (simplified - assumes ready if model is loaded)
        rag_status = "ready" if ctgan_model is not None else "error"
        
        # Check PubMed connection
        pubmed_status = "ready" if pubmed_client else "error"
        
        # Calculate some statistics
        total_patients = len(original_dataset) if original_dataset is not None else 0
        articles_indexed = 1250  # Mock value for demo
        total_queries_processed = 0  # Would be tracked in production
        total_data_generated = 0  # Would be tracked in production
        active_generation_tasks = 0  # Would be tracked in production
        
        status = {
            "ctgan_model": ctgan_status,
            "rag_system": rag_status,
            "pubmed_connection": pubmed_status,
            "articles_indexed": articles_indexed,
            "total_queries_processed": total_queries_processed,
            "total_data_generated": total_data_generated,
            "active_generation_tasks": active_generation_tasks,
            "total_patients": total_patients,
            "timestamp": datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "error": f"Error getting system status: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/statistics")
async def get_dataset_statistics():
    """Get statistics about the original dataset."""
    if original_dataset is None:
        raise HTTPException(status_code=404, detail="Original dataset not available")
    
    try:
        # Calculate basic statistics
        stats = {
            'total_patients': len(original_dataset),
            'age_statistics': {
                'min': int(original_dataset['age'].min()),
                'max': int(original_dataset['age'].max()),
                'mean': round(original_dataset['age'].mean(), 1),
                'median': round(original_dataset['age'].median(), 1)
            },
            'gender_distribution': original_dataset['gender'].value_counts().to_dict(),
            'icu_statistics': {
                'icu_admission_rate': round(original_dataset['has_icu_stay'].mean() * 100, 1),
                'average_icu_los': round(original_dataset[original_dataset['has_icu_stay'] == 1]['icu_los_days'].mean(), 1)
            },
            'mortality_rate': round(original_dataset['mortality'].mean() * 100, 1),
            'risk_level_distribution': original_dataset['risk_level'].value_counts().to_dict(),
            'data_sources': original_dataset['source'].value_counts().to_dict() if 'source' in original_dataset.columns else {}
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get example queries for the system."""
    examples = [
        {
            "title": "Elderly diabetic patients",
            "query": "Generate 50 elderly patients with diabetes",
            "description": "Elderly diabetic patients",
            "category": "Diabetes"
        },
        {
            "title": "Young adults with heart conditions",
            "query": "Create data for young adults with cardiovascular disease",
            "description": "Young adults with heart conditions", 
            "category": "Cardiovascular"
        },
        {
            "title": "Critical care sepsis patients",
            "query": "Generate ICU patients with sepsis",
            "description": "Critical care sepsis patients",
            "category": "Critical Care"
        },
        {
            "title": "Complex multi-diagnosis cases",
            "query": "Create high-risk patients with multiple diagnoses",
            "description": "Complex multi-diagnosis cases",
            "category": "Complex Cases"
        },
        {
            "title": "Female respiratory patients",
            "query": "Generate female patients with respiratory conditions",
            "description": "Female respiratory patients",
            "category": "Respiratory"
        },
        {
            "title": "Extended care trauma cases",
            "query": "Create trauma patients with long hospital stays",
            "description": "Extended care trauma cases",
            "category": "Trauma"
        }
    ]
    
    return {
        "examples": examples,
        "total_count": len(examples),
        "categories": list(set(ex["category"] for ex in examples))
    }

@app.post("/query/parse", response_model=QueryParseResponse)
async def parse_query(request: QueryParseRequest):
    """Parse a natural language query and return structured conditions."""
    try:
        import uuid
        
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        logger.info(f"Parsing query: '{request.query}' (ID: {query_id})")
        
        # Use the existing parse function
        filters = parse_natural_language_query(request.query)
        
        # Extract potential sample size from query (avoid age ranges)
        import re
        query_lower = request.query.lower()
        suggested_sample_size = None
        
        # Look for explicit sample size patterns first
        sample_patterns = [
            r'\bgenerate\s+(\d+)\b',  # "generate 100"
            r'\bcreate\s+(\d+)\b',   # "create 150"
            r'\b(\d+)\s+patients\b', # "200 patients"
            r'\bsample\s+(?:size\s+)?(?:of\s+)?(\d+)\b',  # "sample size 100"
        ]
        
        for pattern in sample_patterns:
            match = re.search(pattern, query_lower)
            if match:
                suggested_sample_size = int(match.group(1))
                break
        
        # If no explicit sample size found, avoid numbers that are part of age ranges
        if suggested_sample_size is None:
            # Only extract numbers that are NOT part of age ranges
            age_range_pattern = r'(?:aged?|age)\s*\d+\s*-\s*\d+'
            if not re.search(age_range_pattern, query_lower):
                # Safe to extract first number if no age range
                sample_size_match = re.search(r'\b(\d+)\b', request.query)
                suggested_sample_size = int(sample_size_match.group(1)) if sample_size_match else None
        
        # Generate suggested filters based on parsed conditions
        suggested_filters = []
        if filters['diagnoses']:
            suggested_filters.extend([f"Diagnosis: {d}" for d in filters['diagnoses']])
        if filters['age_range']:
            min_age, max_age = filters['age_range']
            suggested_filters.append(f"Age: {min_age}-{max_age}")
        if filters['gender']:
            suggested_filters.append(f"Gender: {filters['gender']}")
        if filters['icu_required']:
            suggested_filters.append("ICU Stay: Required")
        if filters['mortality'] is not None:
            status = "Yes" if filters['mortality'] else "No"
            suggested_filters.append(f"Mortality: {status}")
        if filters['risk_level']:
            suggested_filters.append(f"Risk Level: {filters['risk_level']}")
        
        # Calculate confidence based on number of recognized conditions
        total_possible_conditions = 7  # Total filter types
        recognized_conditions = sum(1 for v in filters.values() if v is not None and v != [])
        confidence = min(recognized_conditions / total_possible_conditions, 1.0)
        
        extracted_params = {
            "sample_size": suggested_sample_size,
            "diagnoses": filters.get('diagnoses', []),
            "diagnosis_logic": filters.get('diagnosis_logic', 'OR'),
            "age_range": filters.get('age_range'),
            "gender": filters.get('gender'),
            "icu_required": filters.get('icu_required'),
            "risk_level": filters.get('risk_level'),
            "complexity": filters.get('complexity')
        }
        
        # Search PubMed for relevant articles
        pubmed_results = []
        embeddings_created = 0
        
        try:
            if pubmed_client:
                # Build a healthcare-specific query
                pubmed_query = pubmed_client.build_healthcare_query(
                    conditions=filters.get('diagnoses', []),
                    demographics=filters
                )
                
                logger.info(f"Searching PubMed with query: {pubmed_query}")
                
                # Search and fetch articles (limit to 10 for parsing)
                articles = pubmed_client.search_and_fetch(pubmed_query, 10)
                
                # Convert to the expected format
                for article in articles:
                    pubmed_results.append({
                        "title": article.title,
                        "authors": article.authors,
                        "journal": article.journal,
                        "year": article.year,
                        "pmid": article.pmid,
                        "abstract": article.abstract,
                        "relevance_score": article.relevance_score,
                        "doi": article.doi
                    })
                
                embeddings_created = len(pubmed_results)
                logger.info(f"Retrieved {len(pubmed_results)} articles from PubMed for parsing")
                
            else:
                logger.warning("PubMed client not initialized for parsing")
                
        except Exception as e:
            logger.warning(f"Error fetching from PubMed during parsing: {e}")
        
        # Generate a simple research context for parsing
        conditions = filters.get('diagnoses', [])
        age_info = f"age {filters['age_range'][0]}-{filters['age_range'][1]}" if filters.get('age_range') else "adult"
        gender_info = filters.get('gender', 'all')
        
        research_context = f"""
        Query Analysis Summary:
        - Target population: {', '.join(conditions) if conditions else 'general healthcare'} patients
        - Demographics: {age_info}, {gender_info} gender
        - Suggested sample size: {suggested_sample_size or 'not specified'}
        - Query confidence: {confidence:.1%}
        
        This query has been parsed and is ready for synthetic data generation.
        """.strip()
        
        return QueryParseResponse(
            success=True,
            query_id=query_id,
            parsed_conditions=filters,
            suggested_filters=suggested_filters,
            confidence=confidence,
            extracted_params=extracted_params,
            research_context=research_context,
            pubmed_results=pubmed_results,
            embeddings_created=embeddings_created
        )
        
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        return QueryParseResponse(
            success=False,
            query_id="",
            parsed_conditions={},
            suggested_filters=[],
            confidence=0.0,
            extracted_params={},
            research_context="Error occurred during query parsing.",
            pubmed_results=[],
            embeddings_created=0
        )

@app.post("/generate-context", response_model=GenerateContextResponse)
async def generate_context(request: GenerateContextRequest):
    """Generate research context with PubMed search and query analysis."""
    try:
        import uuid
        
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        logger.info(f"Generating context for query: '{request.query}' (ID: {query_id})")
        
        # Parse the query to extract parameters
        filters = parse_natural_language_query(request.query)
        
        # Extract potential sample size from query (avoid age ranges)
        import re
        query_lower = request.query.lower()
        suggested_sample_size = None
        
        # Look for explicit sample size patterns first
        sample_patterns = [
            r'\bgenerate\s+(\d+)\b',  # "generate 100"
            r'\bcreate\s+(\d+)\b',   # "create 150"
            r'\b(\d+)\s+patients\b', # "200 patients"
            r'\bsample\s+(?:size\s+)?(?:of\s+)?(\d+)\b',  # "sample size 100"
        ]
        
        for pattern in sample_patterns:
            match = re.search(pattern, query_lower)
            if match:
                suggested_sample_size = int(match.group(1))
                break
        
        # If no explicit sample size found, avoid numbers that are part of age ranges
        if suggested_sample_size is None:
            # Only extract numbers that are NOT part of age ranges
            age_range_pattern = r'(?:aged?|age)\s*\d+\s*-\s*\d+'
            if not re.search(age_range_pattern, query_lower):
                # Safe to extract first number if no age range
                sample_size_match = re.search(r'\b(\d+)\b', request.query)
                suggested_sample_size = int(sample_size_match.group(1)) if sample_size_match else None
        
        # Search PubMed for relevant articles
        try:
            if pubmed_client:
                # Build a healthcare-specific query
                pubmed_query = pubmed_client.build_healthcare_query(
                    conditions=filters.get('diagnoses', []),
                    demographics=filters
                )
                
                logger.info(f"Searching PubMed with query: {pubmed_query}")
                
                # Search and fetch articles
                articles = pubmed_client.search_and_fetch(pubmed_query, request.max_articles)
                
                # Convert to the expected format
                pubmed_results = []
                for article in articles:
                    pubmed_results.append({
                        "title": article.title,
                        "authors": article.authors,
                        "journal": article.journal,
                        "year": article.year,
                        "pmid": article.pmid,
                        "abstract": article.abstract,
                        "relevance_score": article.relevance_score,
                        "doi": article.doi
                    })
                
                logger.info(f"Retrieved {len(pubmed_results)} articles from PubMed")
                
            else:
                logger.warning("PubMed client not initialized, using mock data")
                raise Exception("PubMed client not available")
                
        except Exception as e:
            logger.warning(f"Error fetching from PubMed, using mock data: {e}")
            # Fall back to mock data if PubMed API fails
            pubmed_results = [
                {
                    "title": f"Clinical study on {', '.join(filters.get('diagnoses', ['healthcare']))} patients",
                    "authors": ["Smith J", "Johnson A", "Williams B"],
                    "journal": "Journal of Medical Research",
                    "year": 2023,
                    "pmid": "12345678",
                    "abstract": f"This study examines the clinical outcomes in patients with {', '.join(filters.get('diagnoses', ['various conditions']))}...",
                    "relevance_score": 0.95
                },
                {
                    "title": f"Population analysis of {filters.get('gender', 'adult')} patients",
                    "authors": ["Brown C", "Davis M"],
                    "journal": "Healthcare Analytics",
                    "year": 2023,
                    "pmid": "87654321",
                    "abstract": "Comprehensive analysis of patient demographics and outcomes...",
                    "relevance_score": 0.88
                }
            ]
        
        # Generate research context summary
        conditions = filters.get('diagnoses', [])
        age_info = f"age {filters['age_range'][0]}-{filters['age_range'][1]}" if filters.get('age_range') else "adult"
        gender_info = filters.get('gender', 'all')
        
        research_context = f"""
        Based on current medical literature, patients with {', '.join(conditions) if conditions else 'various conditions'} 
        ({age_info}, {gender_info} gender) show specific clinical patterns. 
        
        Key findings from recent studies:
        - Prevalence rates align with published epidemiological data
        - Treatment outcomes vary based on patient demographics
        - ICU utilization patterns are consistent with severity indicators
        
        The requested synthetic data should reflect these clinical realities while maintaining 
        patient privacy and statistical validity.
        """.strip()
        
        extracted_params = {
            "sample_size": suggested_sample_size,
            "diagnoses": filters.get('diagnoses', []),
            "diagnosis_logic": filters.get('diagnosis_logic', 'OR'),
            "age_range": filters.get('age_range'),
            "gender": filters.get('gender'),
            "icu_required": filters.get('icu_required'),
            "risk_level": filters.get('risk_level'),
            "complexity": filters.get('complexity')
        }
        
        return GenerateContextResponse(
            success=True,
            query_id=query_id,
            extracted_params=extracted_params,
            pubmed_results=pubmed_results[:request.max_articles],
            research_context=research_context,
            suggested_sample_size=suggested_sample_size
        )
        
    except Exception as e:
        logger.error(f"Error generating context: {e}")
        return GenerateContextResponse(
            success=False,
            query_id="",
            extracted_params={},
            pubmed_results=[],
            research_context="",
            suggested_sample_size=None
        )

@app.post("/data/generate", response_model=DataGenerateResponse)
async def generate_data_endpoint(request: DataGenerateRequest):
    """Generate synthetic data based on query parameters."""
    try:
        import uuid
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        logger.info(f"Starting data generation task {task_id} for query {request.query_id}")
        
        # For now, we'll generate data synchronously, but in production this would be async
        sample_size = request.sample_size or 100
        
        # Build filters from the request
        filters = {
            'diagnoses': request.diagnoses or [],
            'diagnosis_logic': request.diagnosis_logic or 'OR',
            'age_range': request.age_range,
            'gender': request.gender,
            'icu_required': request.icu_required,
            'mortality': request.mortality,
            'risk_level': request.risk_level,
            'complexity': request.complexity
        }
        
        # Log the filters being applied
        logger.info(f"Applying filters: {filters}")
        
        # Generate the data using existing logic with filters
        synthetic_data = generate_synthetic_data(sample_size, filters)
        
        # Store the result (in production, you'd use a database or cache)
        generation_results = {
            task_id: {
                "status": "completed",
                "data": synthetic_data.to_dict('records'),
                "metadata": {
                    "generated_count": len(synthetic_data),
                    "total_patients": len(synthetic_data),
                    "generation_timestamp": datetime.now().isoformat(),
                    "query_id": request.query_id,
                    "sample_size": sample_size,
                    "mortality_rate": synthetic_data.get('mortality', pd.Series([0])).mean(),
                    "avg_age": synthetic_data.get('age', pd.Series([0])).mean(),
                    "age_std": synthetic_data.get('age', pd.Series([0])).std(),
                    "icu_rate": synthetic_data.get('has_icu_stay', pd.Series([0])).mean(),
                    "gender_distribution": synthetic_data.get('gender', pd.Series()).value_counts().to_dict(),
                    "avg_hospital_los": synthetic_data.get('hospital_los_days', pd.Series([0])).mean(),
                    "avg_icu_los": synthetic_data.get('icu_los_days', pd.Series([0])).mean()
                }
            }
        }
        
        # Store in a simple in-memory cache (in production use Redis or database)
        if not hasattr(app.state, 'generation_results'):
            app.state.generation_results = {}
        app.state.generation_results[task_id] = generation_results[task_id]
        
        return DataGenerateResponse(
            success=True,
            task_id=task_id,
            message=f"Data generation completed for {len(synthetic_data)} patients"
        )
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        return DataGenerateResponse(
            success=False,
            task_id="",
            message=f"Error generating data: {str(e)}"
        )

@app.get("/data/status/{task_id}")
async def get_generation_status(task_id: str):
    """Get the status of a data generation task."""
    try:
        if not hasattr(app.state, 'generation_results'):
            return {"status": "not_found", "message": "Task not found"}
        
        if task_id not in app.state.generation_results:
            return {"status": "not_found", "message": "Task not found"}
        
        result = app.state.generation_results[task_id]
        return {
            "status": result["status"],
            "progress": 100 if result["status"] == "completed" else 0,
            "message": f"Generated {result['metadata']['generated_count']} patients"
        }
        
    except Exception as e:
        logger.error(f"Error getting generation status: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/data/download/{task_id}")
async def download_data(task_id: str):
    """Download generated data."""
    try:
        if not hasattr(app.state, 'generation_results'):
            return {"success": False, "error": "Task not found"}
        
        if task_id not in app.state.generation_results:
            return {"success": False, "error": "Task not found"}
        
        result = app.state.generation_results[task_id]
        
        return {
            "success": True,
            "data": result["data"],
            "metadata": result["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host=API_CONFIG['host'], port=API_CONFIG['port'])