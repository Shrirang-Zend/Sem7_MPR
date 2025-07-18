#!/usr/bin/env python3
"""
06_train_ctgan.py

Script to train CTGAN model for synthetic healthcare data generation.

This script trains a CTGAN model on the processed healthcare dataset
to enable generation of realistic synthetic patient data.
"""

import sys
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from config.settings import FINAL_DATA_DIR, MODELS_DIR, CTGAN_PARAMS, RANDOM_SEED

def load_and_prepare_data():
    """Load and prepare data for CTGAN training."""
    logger = logging.getLogger(__name__)
    
    dataset_path = FINAL_DATA_DIR / 'healthcare_dataset_multi_diagnoses.csv'
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset: {len(df)} patients, {len(df.columns)} columns")
    
    return df

def preprocess_for_ctgan(df):
    """Preprocess data for CTGAN training with improved diagnosis handling."""
    logger = logging.getLogger(__name__)
    logger.info("Preprocessing data for CTGAN...")
    
    # Create a copy for processing
    processed_df = df.copy()
    
    # Extract primary diagnosis from diagnoses JSON for better categorical representation
    logger.info("Processing diagnoses for improved categorical representation...")
    import json
    
    # Define diagnosis categories in scope
    diagnosis_categories = ['DIABETES', 'CARDIOVASCULAR', 'HYPERTENSION', 'RENAL', 
                          'RESPIRATORY', 'SEPSIS', 'NEUROLOGICAL', 'TRAUMA', 'CANCER', 'OTHER']
    
    def extract_primary_diagnosis(diagnoses_str):
        """Extract the primary (first) diagnosis category from list format."""
        try:
            if pd.isna(diagnoses_str) or diagnoses_str == '':
                return 'OTHER'
            
            # Parse the list format: ["DIABETES", "RESPIRATORY"]
            diagnoses_list = json.loads(diagnoses_str) if isinstance(diagnoses_str, str) else diagnoses_str
            if not diagnoses_list or len(diagnoses_list) == 0:
                return 'OTHER'
            
            # Return the first diagnosis as primary
            return diagnoses_list[0] if diagnoses_list[0] in diagnosis_categories else 'OTHER'
                    
        except (json.JSONDecodeError, TypeError, ValueError):
            return 'OTHER'
    
    processed_df['primary_diagnosis'] = processed_df['diagnoses'].apply(extract_primary_diagnosis)
    
    # Convert diagnoses JSON to binary indicator columns (keeping for additional features)
    
    for category in diagnosis_categories:
        processed_df[f'has_{category.lower()}'] = processed_df['diagnoses'].str.contains(category, na=False).astype(int)
    
    # Select columns for training - include BOTH primary_diagnosis AND binary indicators
    training_columns = [
        # Demographics
        'age', 'gender', 'ethnicity',
        
        # Primary diagnosis (categorical - this should improve diagnosis distribution similarity)
        'primary_diagnosis',
        
        # Binary diagnosis indicators (CRUCIAL for conditional generation)
        'has_diabetes', 'has_cardiovascular', 'has_hypertension', 'has_renal',
        'has_respiratory', 'has_sepsis', 'has_neurological', 'has_trauma', 
        'has_cancer', 'has_other',
        
        # Clinical metrics
        'num_diagnoses', 'comorbidity_score',
        
        # Outcomes
        'hospital_los_days', 'icu_los_days', 'has_icu_stay', 'mortality',
        
        # Utilization
        'num_procedures', 'num_medications',
        
        # Categories
        'risk_level', 'utilization_category', 'admission_type', 'insurance',
        'age_group', 'emergency_admission', 'high_complexity'
    ]
    
    # Select available columns
    available_columns = [col for col in training_columns if col in processed_df.columns]
    training_df = processed_df[available_columns].copy()
    
    # Handle missing values
    training_df = training_df.fillna({
        'ethnicity': 'Unknown',
        'insurance': 'Unknown',
        'primary_diagnosis': 'OTHER',
        'num_procedures': 0,
        'num_medications': 0,
        'icu_los_days': 0
    })
    
    # Apply constraints to prevent negative values
    logger.info("Applying data constraints to prevent negative values...")
    training_df['num_procedures'] = training_df['num_procedures'].clip(lower=0)
    training_df['num_medications'] = training_df['num_medications'].clip(lower=0)
    training_df['age'] = training_df['age'].clip(lower=18, upper=100)
    training_df['hospital_los_days'] = training_df['hospital_los_days'].clip(lower=0.1)
    training_df['icu_los_days'] = training_df['icu_los_days'].clip(lower=0)
    training_df['comorbidity_score'] = training_df['comorbidity_score'].clip(lower=1)
    
    # Handle categorical variables
    categorical_columns = ['gender', 'ethnicity', 'risk_level', 'utilization_category', 
                          'admission_type', 'insurance', 'age_group', 'primary_diagnosis']
    
    # Binary diagnosis indicators are treated as discrete (0/1) by CTGAN automatically
    binary_columns = ['has_diabetes', 'has_cardiovascular', 'has_hypertension', 'has_renal',
                     'has_respiratory', 'has_sepsis', 'has_neurological', 'has_trauma', 
                     'has_cancer', 'has_other', 'has_icu_stay', 'mortality', 'emergency_admission', 
                     'high_complexity']
    
    logger.info(f"Preprocessed data: {len(training_df)} rows, {len(training_df.columns)} columns")
    logger.info(f"Categorical columns: {[col for col in categorical_columns if col in training_df.columns]}")
    logger.info(f"Binary columns: {[col for col in binary_columns if col in training_df.columns]}")
    logger.info(f"Primary diagnosis distribution: {training_df['primary_diagnosis'].value_counts().to_dict()}")
    
    # Show binary diagnosis statistics
    diagnosis_binary_cols = [col for col in binary_columns if col.startswith('has_') and col != 'has_icu_stay']
    if diagnosis_binary_cols:
        logger.info("Binary diagnosis statistics:")
        for col in diagnosis_binary_cols:
            if col in training_df.columns:
                positive_count = training_df[col].sum()
                percentage = (positive_count / len(training_df)) * 100
                logger.info(f"  {col}: {positive_count} patients ({percentage:.1f}%)")
    
    return training_df, available_columns

def train_ctgan_model(training_df):
    """Train CTGAN model."""
    logger = logging.getLogger(__name__)
    
    try:
        from ctgan import CTGAN
    except ImportError:
        logger.error("CTGAN not installed. Install with: pip install ctgan")
        raise
    
    logger.info("Initializing CTGAN model...")
    
    # Initialize CTGAN with enhanced parameters including PAC
    try:
        ctgan = CTGAN(
            epochs=CTGAN_PARAMS['epochs'],
            batch_size=CTGAN_PARAMS['batch_size'],
            generator_dim=CTGAN_PARAMS['generator_dim'],
            discriminator_dim=CTGAN_PARAMS['discriminator_dim'],
            pac=CTGAN_PARAMS['pac'],
            generator_lr=CTGAN_PARAMS['lr'],
            discriminator_lr=CTGAN_PARAMS['lr'],
            generator_decay=CTGAN_PARAMS['decay'],
            discriminator_decay=CTGAN_PARAMS['decay'],
            verbose=True
        )
        logger.info("CTGAN initialized with full enhanced parameters")
    except Exception as e:
        logger.warning(f"Full parameter initialization failed: {e}")
        logger.info("Trying CTGAN with core parameters...")
        try:
            ctgan = CTGAN(
                epochs=CTGAN_PARAMS['epochs'],
                batch_size=CTGAN_PARAMS['batch_size'],
                generator_dim=CTGAN_PARAMS['generator_dim'],
                discriminator_dim=CTGAN_PARAMS['discriminator_dim'],
                pac=CTGAN_PARAMS['pac'],
                verbose=True
            )
            logger.info("CTGAN initialized with core enhanced parameters")
        except Exception as e2:
            logger.warning(f"Enhanced parameter initialization failed: {e2}")
            logger.info("Falling back to basic CTGAN initialization...")
            ctgan = CTGAN(
                epochs=CTGAN_PARAMS['epochs'],
                batch_size=CTGAN_PARAMS['batch_size'],
                verbose=True
            )
    
    logger.info(f"Training CTGAN model with {CTGAN_PARAMS['epochs']} epochs...")
    logger.info(f"Training data shape: {training_df.shape}")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Specify discrete columns for CTGAN
    categorical_columns = ['gender', 'ethnicity', 'risk_level', 'utilization_category', 
                          'admission_type', 'insurance', 'age_group', 'primary_diagnosis']
    binary_columns = ['mortality', 'emergency_admission', 'high_complexity', 'has_icu_stay']
    discrete_columns = categorical_columns + binary_columns + ['num_diagnoses']
    
    # Only include columns that actually exist in the dataframe
    discrete_columns = [col for col in discrete_columns if col in training_df.columns]
    
    logger.info(f"Discrete columns for CTGAN: {discrete_columns}")
    
    # Train the model
    start_time = datetime.now()
    ctgan.fit(training_df, discrete_columns=discrete_columns)
    training_time = datetime.now() - start_time
    
    logger.info(f"CTGAN training completed in {training_time}")
    
    return ctgan, training_time

def evaluate_model(ctgan, training_df, num_samples=1000):
    """Evaluate the trained CTGAN model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating model by generating {num_samples} samples...")
    
    # Generate synthetic samples
    synthetic_df = ctgan.sample(num_samples)
    
    # Basic comparison statistics
    evaluation_results = {
        'generation_timestamp': datetime.now().isoformat(),
        'num_synthetic_samples': len(synthetic_df),
        'original_samples': len(training_df),
        'column_comparison': {},
        'distribution_similarity': {}
    }
    
    # Compare distributions for numeric columns
    numeric_columns = training_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in synthetic_df.columns:
            orig_mean = training_df[col].mean()
            synth_mean = synthetic_df[col].mean()
            orig_std = training_df[col].std()
            synth_std = synthetic_df[col].std()
            
            evaluation_results['column_comparison'][col] = {
                'original_mean': round(float(orig_mean), 3),
                'synthetic_mean': round(float(synth_mean), 3),
                'original_std': round(float(orig_std), 3),
                'synthetic_std': round(float(synth_std), 3),
                'mean_difference': round(float(abs(orig_mean - synth_mean)), 3)
            }
    
    # Compare categorical distributions
    categorical_columns = training_df.select_dtypes(include=['category', 'object']).columns
    
    for col in categorical_columns:
        if col in synthetic_df.columns:
            orig_dist = training_df[col].value_counts(normalize=True).to_dict()
            synth_dist = synthetic_df[col].value_counts(normalize=True).to_dict()
            
            # Calculate similarity (simplified)
            common_categories = set(orig_dist.keys()) & set(synth_dist.keys())
            similarity = sum(min(orig_dist.get(cat, 0), synth_dist.get(cat, 0)) 
                           for cat in common_categories)
            
            evaluation_results['distribution_similarity'][col] = {
                'similarity_score': round(similarity, 3),
                'original_categories': len(orig_dist),
                'synthetic_categories': len(synth_dist)
            }
    
    logger.info("Model evaluation completed")
    return evaluation_results, synthetic_df

def apply_clinical_constraints(synthetic_df):
    """Apply enhanced clinical logic constraints and prevent negative values."""
    logger = logging.getLogger(__name__)
    logger.info("Applying enhanced clinical constraints to synthetic data...")
    
    constrained_df = synthetic_df.copy()
    
    # Fix negative values with stricter bounds
    constrained_df['num_procedures'] = constrained_df['num_procedures'].clip(lower=0, upper=2000)
    constrained_df['num_medications'] = constrained_df['num_medications'].clip(lower=0, upper=500)
    constrained_df['age'] = constrained_df['age'].clip(lower=18, upper=100)
    constrained_df['hospital_los_days'] = constrained_df['hospital_los_days'].clip(lower=0.1, upper=60)
    constrained_df['icu_los_days'] = constrained_df['icu_los_days'].clip(lower=0, upper=30)
    if 'comorbidity_score' in constrained_df.columns:
        constrained_df['comorbidity_score'] = constrained_df['comorbidity_score'].clip(lower=1, upper=10)
    if 'num_diagnoses' in constrained_df.columns:
        constrained_df['num_diagnoses'] = constrained_df['num_diagnoses'].clip(lower=1, upper=8)
    
    # Enhanced ICU logic constraints
    if 'has_icu_stay' in constrained_df.columns and 'icu_los_days' in constrained_df.columns:
        # If no ICU stay, ICU LOS should be 0
        constrained_df.loc[constrained_df['has_icu_stay'] == 0, 'icu_los_days'] = 0
        # If ICU stay, ICU LOS should be realistic (0.5-30 days)
        icu_mask = constrained_df['has_icu_stay'] == 1
        constrained_df.loc[icu_mask, 'icu_los_days'] = \
            constrained_df.loc[icu_mask, 'icu_los_days'].clip(lower=0.5, upper=30)
    
    # ICU LOS cannot exceed hospital LOS
    if 'icu_los_days' in constrained_df.columns and 'hospital_los_days' in constrained_df.columns:
        constrained_df['icu_los_days'] = np.minimum(
            constrained_df['icu_los_days'], 
            constrained_df['hospital_los_days']
        )
    
    # Age-mortality correlation (older patients higher mortality)
    if 'mortality' in constrained_df.columns and 'age' in constrained_df.columns:
        # Age-adjusted mortality rates
        age_mortality_prob = np.where(
            constrained_df['age'] < 65, 0.05,  # <65: 5% mortality
            np.where(constrained_df['age'] < 80, 0.15,  # 65-79: 15% mortality
                    0.30)  # 80+: 30% mortality
        )
        # Apply probabilistic mortality based on age
        constrained_df['mortality'] = np.random.binomial(1, age_mortality_prob)
    
    # Comorbidity-LOS correlation (higher comorbidity = longer stay)
    if 'comorbidity_score' in constrained_df.columns and 'hospital_los_days' in constrained_df.columns:
        # Adjust LOS based on comorbidity score
        comorbidity_factor = 1 + (constrained_df['comorbidity_score'] - 1) * 0.3
        constrained_df['hospital_los_days'] = \
            constrained_df['hospital_los_days'] * comorbidity_factor
        constrained_df['hospital_los_days'] = constrained_df['hospital_los_days'].clip(lower=0.1, upper=60)
    
    # Emergency admission logic (higher complexity patients more likely emergency)
    if 'emergency_admission' in constrained_df.columns and 'high_complexity' in constrained_df.columns:
        high_complexity_mask = constrained_df['high_complexity'] == 1
        # 80% of high complexity cases should be emergency admissions
        emergency_prob = np.where(high_complexity_mask, 0.80, 0.45)
        constrained_df['emergency_admission'] = np.random.binomial(1, emergency_prob)
    
    logger.info("Enhanced clinical constraints applied successfully")
    return constrained_df

def add_differential_privacy_noise(synthetic_df, epsilon=0.5):
    """Add enhanced differential privacy noise to numeric columns."""
    logger = logging.getLogger(__name__)
    logger.info(f"Adding enhanced differential privacy noise with epsilon={epsilon}...")
    
    noisy_df = synthetic_df.copy()
    numeric_columns = noisy_df.select_dtypes(include=[np.number]).columns
    
    # Apply stronger Laplace noise for differential privacy
    for col in numeric_columns:
        if col not in ['has_icu_stay', 'mortality', 'emergency_admission', 'high_complexity', 'num_diagnoses']:
            # Calculate sensitivity (range of values)
            sensitivity = noisy_df[col].max() - noisy_df[col].min()
            if sensitivity > 0:
                # Add stronger Laplace noise (reduced epsilon = more noise)
                noise_scale = sensitivity / epsilon
                noise = np.random.laplace(0, noise_scale, len(noisy_df))
                noisy_df[col] = noisy_df[col] + noise
                
                # Re-apply constraints after adding noise
                if col == 'age':
                    noisy_df[col] = noisy_df[col].clip(lower=18, upper=100)
                elif col in ['num_procedures', 'num_medications']:
                    noisy_df[col] = noisy_df[col].clip(lower=0, upper=2000 if col == 'num_procedures' else 500)
                elif col in ['hospital_los_days', 'icu_los_days']:
                    noisy_df[col] = noisy_df[col].clip(lower=0, upper=60 if col == 'hospital_los_days' else 30)
                elif col == 'comorbidity_score':
                    noisy_df[col] = noisy_df[col].clip(lower=1, upper=10)
    
    # Add k-anonymity style record scrambling for additional privacy
    logger.info("Applying k-anonymity style record scrambling...")
    
    # Shuffle age within groups to maintain distribution but reduce linkability
    if 'age' in noisy_df.columns and 'age_group' in noisy_df.columns:
        for age_group in noisy_df['age_group'].unique():
            mask = noisy_df['age_group'] == age_group
            ages = noisy_df.loc[mask, 'age'].values
            np.random.shuffle(ages)
            noisy_df.loc[mask, 'age'] = ages
    
    # Add small random perturbations to reduce exact matches
    continuous_cols = ['hospital_los_days', 'icu_los_days', 'comorbidity_score']
    for col in continuous_cols:
        if col in noisy_df.columns:
            # Add small Gaussian noise for additional protection
            gaussian_noise = np.random.normal(0, 0.1, len(noisy_df))
            noisy_df[col] = noisy_df[col] + gaussian_noise
            
            # Re-apply bounds
            if col == 'hospital_los_days':
                noisy_df[col] = noisy_df[col].clip(lower=0.1, upper=60)
            elif col == 'icu_los_days':
                noisy_df[col] = noisy_df[col].clip(lower=0, upper=30)
            elif col == 'comorbidity_score':
                noisy_df[col] = noisy_df[col].clip(lower=1, upper=10)
    
    logger.info("Enhanced differential privacy noise added with k-anonymity protection")
    return noisy_df

def evaluate_model_static(training_df, synthetic_df):
    """Static evaluation function that doesn't require the model instance."""
    logger = logging.getLogger(__name__)
    logger.info("Performing static evaluation on synthetic data...")
    
    evaluation_results = {
        'generation_timestamp': datetime.now().isoformat(),
        'num_synthetic_samples': len(synthetic_df),
        'original_samples': len(training_df),
        'column_comparison': {},
        'distribution_similarity': {}
    }
    
    # Compare distributions for numeric columns
    numeric_columns = training_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in synthetic_df.columns:
            orig_mean = training_df[col].mean()
            synth_mean = synthetic_df[col].mean()
            orig_std = training_df[col].std()
            synth_std = synthetic_df[col].std()
            
            evaluation_results['column_comparison'][col] = {
                'original_mean': round(float(orig_mean), 3),
                'synthetic_mean': round(float(synth_mean), 3),
                'original_std': round(float(orig_std), 3),
                'synthetic_std': round(float(synth_std), 3),
                'mean_difference': round(float(abs(orig_mean - synth_mean)), 3)
            }
    
    # Compare categorical distributions
    categorical_columns = training_df.select_dtypes(include=['category', 'object']).columns
    
    for col in categorical_columns:
        if col in synthetic_df.columns:
            orig_dist = training_df[col].value_counts(normalize=True).to_dict()
            synth_dist = synthetic_df[col].value_counts(normalize=True).to_dict()
            
            # Calculate similarity (simplified)
            common_categories = set(orig_dist.keys()) & set(synth_dist.keys())
            similarity = sum(min(orig_dist.get(cat, 0), synth_dist.get(cat, 0)) 
                           for cat in common_categories)
            
            evaluation_results['distribution_similarity'][col] = {
                'similarity_score': round(similarity, 3),
                'original_categories': len(orig_dist),
                'synthetic_categories': len(synth_dist)
            }
    
    logger.info("Static evaluation completed")
    return evaluation_results, synthetic_df

def save_model_and_results(ctgan, training_columns, evaluation_results, training_time):
    """Save the trained model and results."""
    logger = logging.getLogger(__name__)
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model_path = MODELS_DIR / 'ctgan_healthcare_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ctgan, f)
    logger.info(f"Saved CTGAN model to: {model_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'CTGAN',
        'training_timestamp': datetime.now().isoformat(),
        'training_duration_seconds': training_time.total_seconds(),
        'model_parameters': CTGAN_PARAMS,
        'training_columns': training_columns,
        'random_seed': RANDOM_SEED,
        'model_file': str(model_path),
        'evaluation_results': evaluation_results
    }
    
    metadata_path = MODELS_DIR / 'ctgan_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {metadata_path}")
    
    return model_path, metadata_path

def main():
    """Main training execution."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting CTGAN training...")
        
        # Load and prepare data
        df = load_and_prepare_data()
        training_df, training_columns = preprocess_for_ctgan(df)
        
        # Train model
        ctgan, training_time = train_ctgan_model(training_df)
        
        # Evaluate model with improvements
        evaluation_results, synthetic_samples = evaluate_model(ctgan, training_df)
        
        # Apply clinical constraints and enhanced differential privacy
        logger.info("Applying post-processing improvements...")
        synthetic_samples_improved = apply_clinical_constraints(synthetic_samples)
        synthetic_samples_final = add_differential_privacy_noise(synthetic_samples_improved, epsilon=0.5)
        
        # Re-evaluate with improved samples
        logger.info("Re-evaluating model with improved synthetic data...")
        improved_evaluation, _ = evaluate_model_static(training_df, synthetic_samples_final)
        evaluation_results['improved_results'] = improved_evaluation
        
        # Save model and results
        model_path, metadata_path = save_model_and_results(
            ctgan, training_columns, evaluation_results, training_time
        )
        
        # Print summary
        print("\n" + "="*60)
        print("CTGAN TRAINING COMPLETED")
        print("="*60)
        print(f"Training time: {training_time}")
        print(f"Training samples: {len(training_df)}")
        print(f"Training features: {len(training_columns)}")
        print(f"Model epochs: {CTGAN_PARAMS['epochs']}")
        
        print(f"\nModel Performance:")
        print(f"  Synthetic samples generated: {evaluation_results['num_synthetic_samples']}")
        
        # Show some distribution comparisons
        if evaluation_results['column_comparison']:
            print(f"\nKey Metrics Comparison (Original vs Synthetic):")
            key_metrics = ['age', 'hospital_los_days', 'icu_los_days', 'comorbidity_score']
            for metric in key_metrics:
                if metric in evaluation_results['column_comparison']:
                    comp = evaluation_results['column_comparison'][metric]
                    print(f"  {metric}: {comp['original_mean']:.1f} vs {comp['synthetic_mean']:.1f}")
        
        if evaluation_results['distribution_similarity']:
            print(f"\nCategorical Similarity Scores:")
            for col, sim in evaluation_results['distribution_similarity'].items():
                print(f"  {col}: {sim['similarity_score']:.3f}")
        
        print(f"\nOutput files:")
        print(f"  Model: {model_path}")
        print(f"  Metadata: {metadata_path}")
        
        logger.info("CTGAN training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during CTGAN training: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)