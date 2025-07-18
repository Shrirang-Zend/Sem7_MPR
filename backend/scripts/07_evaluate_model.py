#!/usr/bin/env python3
"""
07_evaluate_model.py

Script to evaluate the trained CTGAN model performance.

This script provides comprehensive evaluation of the CTGAN model including
distribution similarity, clinical validity, and generation quality metrics.
"""

import sys
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from config.settings import MODELS_DIR, FINAL_DATA_DIR, LOGS_DIR
from src.utils.common_utils import safe_json_loads

class CTGANModelEvaluator:
    """
    Comprehensive evaluation of CTGAN model performance.
    """
    
    def __init__(self):
        self.model = None
        self.original_data = None
        self.synthetic_data = None
        self.evaluation_results = {}
        
    def load_model_and_data(self):
        """Load the trained model and original dataset."""
        logger = logging.getLogger(__name__)
        
        # Load CTGAN model
        model_path = MODELS_DIR / 'ctgan_healthcare_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"CTGAN model not found: {model_path}")
        
        logger.info(f"Loading CTGAN model from {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load original dataset
        dataset_path = FINAL_DATA_DIR / 'healthcare_dataset_multi_diagnoses.csv'
        if not dataset_path.exists():
            raise FileNotFoundError(f"Original dataset not found: {dataset_path}")
        
        logger.info(f"Loading original dataset from {dataset_path}")
        self.original_data = pd.read_csv(dataset_path)
        
        logger.info(f"Loaded model and original data: {len(self.original_data)} patients")
    
    def generate_synthetic_data(self, num_samples: int = None):
        """Generate synthetic data for evaluation."""
        logger = logging.getLogger(__name__)
        
        if num_samples is None:
            num_samples = len(self.original_data)
        
        logger.info(f"Generating {num_samples} synthetic patients for evaluation")
        
        # Generate synthetic data
        self.synthetic_data = self.model.sample(num_samples)
        
        # Post-process synthetic data
        self.synthetic_data = self._post_process_synthetic_data(self.synthetic_data)
        
        logger.info(f"Generated {len(self.synthetic_data)} synthetic patients")
    
    def _post_process_synthetic_data(self, df):
        """Post-process synthetic data for evaluation."""
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
        
        # Add primary_diagnosis if missing in synthetic data but present in original
        if 'primary_diagnosis' not in processed_df.columns and 'primary_diagnosis' in self.original_data.columns:
            # Try to infer primary diagnosis from binary columns or other indicators
            processed_df = self._infer_primary_diagnosis(processed_df, binary_columns)
        
        return processed_df
    
    def _infer_primary_diagnosis(self, df, binary_columns):
        """Infer primary diagnosis from binary indicators for synthetic data."""
        logger = logging.getLogger(__name__)
        logger.info("Inferring primary diagnosis from binary indicators...")
        
        primary_diagnoses = []
        diagnosis_categories = ['DIABETES', 'CARDIOVASCULAR', 'HYPERTENSION', 'RENAL', 
                              'RESPIRATORY', 'SEPSIS', 'NEUROLOGICAL', 'TRAUMA', 'CANCER', 'OTHER']
        
        for _, row in df.iterrows():
            # Find first active diagnosis
            primary = 'OTHER'
            for category in diagnosis_categories:
                col_name = f'has_{category.lower()}'
                if col_name in row and row.get(col_name, 0) == 1:
                    primary = category
                    break
            primary_diagnoses.append(primary)
        
        df['primary_diagnosis'] = primary_diagnoses
        logger.info(f"Inferred primary diagnosis distribution: {pd.Series(primary_diagnoses).value_counts().to_dict()}")
        return df
    
    def evaluate_statistical_similarity(self):
        """Evaluate statistical similarity between original and synthetic data."""
        logger = logging.getLogger(__name__)
        logger.info("Evaluating statistical similarity...")
        
        results = {
            'numeric_distributions': {},
            'categorical_distributions': {},
            'correlation_analysis': {},
            'summary_statistics': {}
        }
        
        # Get common columns
        common_columns = list(set(self.original_data.columns) & set(self.synthetic_data.columns))
        
        # Evaluate numeric columns
        numeric_columns = self.original_data[common_columns].select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in self.synthetic_data.columns:
                original_values = self.original_data[col].dropna()
                synthetic_values = self.synthetic_data[col].dropna()
                
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.kstest(synthetic_values, original_values.values)
                
                # Summary statistics comparison
                orig_stats = {
                    'mean': float(original_values.mean()),
                    'std': float(original_values.std()),
                    'median': float(original_values.median()),
                    'min': float(original_values.min()),
                    'max': float(original_values.max())
                }
                
                synth_stats = {
                    'mean': float(synthetic_values.mean()),
                    'std': float(synthetic_values.std()),
                    'median': float(synthetic_values.median()),
                    'min': float(synthetic_values.min()),
                    'max': float(synthetic_values.max())
                }
                
                results['numeric_distributions'][col] = {
                    'ks_statistic': float(ks_statistic),
                    'ks_p_value': float(ks_p_value),
                    'similarity_score': 1 - ks_statistic,  # Higher is better
                    'original_stats': orig_stats,
                    'synthetic_stats': synth_stats,
                    'mean_difference': abs(orig_stats['mean'] - synth_stats['mean']),
                    'std_difference': abs(orig_stats['std'] - synth_stats['std'])
                }
        
        # Evaluate categorical columns
        categorical_columns = self.original_data[common_columns].select_dtypes(include=['object', 'category']).columns
        
        # Ensure primary_diagnosis is included if it exists in both datasets
        if 'primary_diagnosis' in self.original_data.columns and 'primary_diagnosis' in self.synthetic_data.columns:
            categorical_columns = categorical_columns.tolist() + ['primary_diagnosis']
            categorical_columns = list(set(categorical_columns))  # Remove duplicates
        
        for col in categorical_columns:
            if col in self.synthetic_data.columns and col != 'patient_id':
                orig_dist = self.original_data[col].value_counts(normalize=True)
                synth_dist = self.synthetic_data[col].value_counts(normalize=True)
                
                # Calculate Total Variation Distance
                all_categories = set(orig_dist.index) | set(synth_dist.index)
                tvd = 0.5 * sum(abs(orig_dist.get(cat, 0) - synth_dist.get(cat, 0)) for cat in all_categories)
                
                results['categorical_distributions'][col] = {
                    'total_variation_distance': float(tvd),
                    'similarity_score': 1 - tvd,  # Higher is better
                    'original_categories': len(orig_dist),
                    'synthetic_categories': len(synth_dist),
                    'common_categories': len(set(orig_dist.index) & set(synth_dist.index))
                }
        
        # Correlation analysis for numeric columns
        if len(numeric_columns) > 1:
            orig_corr = self.original_data[numeric_columns].corr()
            synth_corr = self.synthetic_data[numeric_columns].corr()
            
            # Flatten correlation matrices and compare
            orig_corr_flat = orig_corr.values[np.triu_indices_from(orig_corr.values, k=1)]
            synth_corr_flat = synth_corr.values[np.triu_indices_from(synth_corr.values, k=1)]
            
            corr_similarity = stats.pearsonr(orig_corr_flat, synth_corr_flat)[0]
            
            results['correlation_analysis'] = {
                'correlation_similarity': float(corr_similarity),
                'original_mean_correlation': float(np.mean(np.abs(orig_corr_flat))),
                'synthetic_mean_correlation': float(np.mean(np.abs(synth_corr_flat)))
            }
        
        # Overall summary
        numeric_similarities = [r['similarity_score'] for r in results['numeric_distributions'].values()]
        categorical_similarities = [r['similarity_score'] for r in results['categorical_distributions'].values()]
        
        results['summary_statistics'] = {
            'overall_numeric_similarity': float(np.mean(numeric_similarities)) if numeric_similarities else 0.0,
            'overall_categorical_similarity': float(np.mean(categorical_similarities)) if categorical_similarities else 0.0,
            'total_columns_evaluated': len(numeric_similarities) + len(categorical_similarities)
        }
        
        self.evaluation_results['statistical_similarity'] = results
        logger.info("Statistical similarity evaluation completed")
    
    def evaluate_clinical_validity(self):
        """Evaluate clinical validity and logical consistency."""
        logger = logging.getLogger(__name__)
        logger.info("Evaluating clinical validity...")
        
        results = {
            'age_distribution': {},
            'diagnosis_patterns': {},
            'clinical_logic': {},
            'mortality_patterns': {},
            'icu_patterns': {}
        }
        
        # Age distribution analysis
        orig_age = self.original_data['age']
        synth_age = self.synthetic_data['age']
        
        results['age_distribution'] = {
            'original_mean_age': float(orig_age.mean()),
            'synthetic_mean_age': float(synth_age.mean()),
            'age_difference': float(abs(orig_age.mean() - synth_age.mean())),
            'original_age_range': [int(orig_age.min()), int(orig_age.max())],
            'synthetic_age_range': [int(synth_age.min()), int(synth_age.max())],
            'adults_only_original': bool((orig_age >= 18).all()),
            'adults_only_synthetic': bool((synth_age >= 18).all())
        }
        
        # Primary diagnosis patterns analysis (improved approach)
        if 'primary_diagnosis' in self.original_data.columns and 'primary_diagnosis' in self.synthetic_data.columns:
            orig_diagnoses = self.original_data['primary_diagnosis'].value_counts(normalize=True)
            synth_diagnoses = self.synthetic_data['primary_diagnosis'].value_counts(normalize=True)
            
            # Calculate diagnosis distribution similarity using Total Variation Distance
            all_diagnoses = set(orig_diagnoses.index) | set(synth_diagnoses.index)
            diagnosis_tvd = 0.5 * sum(abs(orig_diagnoses.get(dx, 0) - synth_diagnoses.get(dx, 0)) 
                                    for dx in all_diagnoses)
            
            common_diagnoses = set(orig_diagnoses.index) & set(synth_diagnoses.index)
            
            results['diagnosis_patterns'] = {
                'diagnosis_distribution_similarity': 1 - diagnosis_tvd,
                'original_unique_diagnoses': len(orig_diagnoses),
                'synthetic_unique_diagnoses': len(synth_diagnoses),
                'common_diagnoses': len(common_diagnoses),
                'top_diagnoses_original': orig_diagnoses.head(5).to_dict(),
                'top_diagnoses_synthetic': synth_diagnoses.head(5).to_dict()
            }
        else:
            # Fallback to old diagnoses field if primary_diagnosis not available
            results['diagnosis_patterns'] = {
                'diagnosis_distribution_similarity': 0.0,
                'original_unique_diagnoses': 0,
                'synthetic_unique_diagnoses': 0,
                'common_diagnoses': 0,
                'top_diagnoses_original': {},
                'top_diagnoses_synthetic': {},
                'note': 'primary_diagnosis field not found in both datasets'
            }
        
        # Clinical logic checks
        clinical_checks = {}
        
        # ICU LOS consistency
        if 'has_icu_stay' in self.synthetic_data.columns and 'icu_los_days' in self.synthetic_data.columns:
            icu_patients = self.synthetic_data[self.synthetic_data['has_icu_stay'] == 1]
            non_icu_patients = self.synthetic_data[self.synthetic_data['has_icu_stay'] == 0]
            
            clinical_checks['icu_los_consistency'] = {
                'icu_patients_have_los': bool((icu_patients['icu_los_days'] > 0).all()) if len(icu_patients) > 0 else True,
                'non_icu_patients_zero_los': bool((non_icu_patients['icu_los_days'] == 0).all()) if len(non_icu_patients) > 0 else True
            }
        
        # Hospital LOS >= ICU LOS
        if 'hospital_los_days' in self.synthetic_data.columns and 'icu_los_days' in self.synthetic_data.columns:
            clinical_checks['hospital_icu_los_relationship'] = bool(
                (self.synthetic_data['hospital_los_days'] >= self.synthetic_data['icu_los_days']).all()
            )
        
        # Age and complexity relationship
        if 'age' in self.synthetic_data.columns and 'comorbidity_score' in self.synthetic_data.columns:
            age_complexity_corr = self.synthetic_data['age'].corr(self.synthetic_data['comorbidity_score'])
            clinical_checks['age_complexity_correlation'] = {
                'correlation': float(age_complexity_corr),
                'positive_correlation': bool(age_complexity_corr > 0)
            }
        
        results['clinical_logic'] = clinical_checks
        
        # Mortality patterns
        if 'mortality' in self.synthetic_data.columns:
            orig_mortality = self.original_data['mortality'].mean()
            synth_mortality = self.synthetic_data['mortality'].mean()
            
            results['mortality_patterns'] = {
                'original_mortality_rate': float(orig_mortality),
                'synthetic_mortality_rate': float(synth_mortality),
                'mortality_rate_difference': float(abs(orig_mortality - synth_mortality)),
                'realistic_mortality_range': bool(0.05 <= synth_mortality <= 0.30)  # Reasonable ICU mortality range
            }
        
        # ICU patterns
        if 'has_icu_stay' in self.synthetic_data.columns:
            orig_icu_rate = self.original_data['has_icu_stay'].mean()
            synth_icu_rate = self.synthetic_data['has_icu_stay'].mean()
            
            results['icu_patterns'] = {
                'original_icu_rate': float(orig_icu_rate),
                'synthetic_icu_rate': float(synth_icu_rate),
                'icu_rate_difference': float(abs(orig_icu_rate - synth_icu_rate)),
                'realistic_icu_range': bool(0.15 <= synth_icu_rate <= 0.50)  # Reasonable ICU admission range
            }
        
        self.evaluation_results['clinical_validity'] = results
        logger.info("Clinical validity evaluation completed")
    
    def _extract_all_diagnoses(self, diagnoses_series):
        """Extract all diagnoses from JSON-encoded diagnoses column."""
        all_diagnoses = []
        for diagnoses_json in diagnoses_series:
            try:
                diagnoses = safe_json_loads(diagnoses_json, [])
                all_diagnoses.extend(diagnoses)
            except:
                continue
        return all_diagnoses
    
    def evaluate_utility_preservation(self):
        """Evaluate how well the synthetic data preserves utility for ML tasks."""
        logger = logging.getLogger(__name__)
        logger.info("Evaluating utility preservation...")
        
        results = {}
        
        try:
            # Prepare data for ML evaluation
            orig_ml_data, orig_target = self._prepare_ml_data(self.original_data)
            synth_ml_data, synth_target = self._prepare_ml_data(self.synthetic_data)
            
            if orig_ml_data is not None and synth_ml_data is not None:
                # Train on original, test on original (baseline)
                orig_train_x, orig_test_x, orig_train_y, orig_test_y = train_test_split(
                    orig_ml_data, orig_target, test_size=0.3, random_state=42
                )
                
                # Train on synthetic, test on original (utility test)
                synth_train_x, synth_train_y = synth_ml_data, synth_target
                
                # Train models
                rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
                
                rf_orig.fit(orig_train_x, orig_train_y)
                rf_synth.fit(synth_train_x, synth_train_y)
                
                # Evaluate on original test set
                orig_pred = rf_orig.predict(orig_test_x)
                synth_pred = rf_synth.predict(orig_test_x)
                
                # Calculate utility scores
                from sklearn.metrics import accuracy_score, f1_score
                
                orig_accuracy = accuracy_score(orig_test_y, orig_pred)
                synth_accuracy = accuracy_score(orig_test_y, synth_pred)
                
                orig_f1 = f1_score(orig_test_y, orig_pred, average='weighted')
                synth_f1 = f1_score(orig_test_y, synth_pred, average='weighted')
                
                results['ml_utility'] = {
                    'original_model_accuracy': float(orig_accuracy),
                    'synthetic_model_accuracy': float(synth_accuracy),
                    'accuracy_ratio': float(synth_accuracy / orig_accuracy),
                    'original_model_f1': float(orig_f1),
                    'synthetic_model_f1': float(synth_f1),
                    'f1_ratio': float(synth_f1 / orig_f1),
                    'utility_preservation_score': float((synth_accuracy / orig_accuracy + synth_f1 / orig_f1) / 2)
                }
                
        except Exception as e:
            logger.warning(f"Could not evaluate ML utility: {e}")
            results['ml_utility'] = {'error': str(e)}
        
        self.evaluation_results['utility_preservation'] = results
        logger.info("Utility preservation evaluation completed")
    
    def _prepare_ml_data(self, df):
        """Prepare data for machine learning evaluation."""
        if 'mortality' not in df.columns:
            return None, None
        
        # Select features for ML
        feature_columns = []
        for col in df.columns:
            if col in ['age', 'hospital_los_days', 'icu_los_days', 'comorbidity_score', 'num_diagnoses']:
                feature_columns.append(col)
            elif col.startswith('has_'):
                feature_columns.append(col)
        
        if not feature_columns:
            return None, None
        
        X = df[feature_columns].fillna(0)
        y = df['mortality'].astype(int)
        
        return X, y
    
    def evaluate_privacy_protection(self):
        """Evaluate privacy protection of synthetic data."""
        logger = logging.getLogger(__name__)
        logger.info("Evaluating privacy protection...")
        
        results = {
            'distance_to_closest_record': {},
            'membership_inference': {},
            'attribute_disclosure': {}
        }
        
        # Distance to closest record analysis
        try:
            # Sample subset for computational efficiency
            sample_size = min(500, len(self.original_data), len(self.synthetic_data))
            orig_sample = self.original_data.sample(sample_size, random_state=42)
            synth_sample = self.synthetic_data.sample(sample_size, random_state=42)
            
            # Get numeric columns for distance calculation
            numeric_cols = orig_sample.select_dtypes(include=[np.number]).columns
            common_numeric = [col for col in numeric_cols if col in synth_sample.columns]
            
            if common_numeric:
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics.pairwise import euclidean_distances
                
                # Standardize features
                scaler = StandardScaler()
                orig_scaled = scaler.fit_transform(orig_sample[common_numeric].fillna(0))
                synth_scaled = scaler.transform(synth_sample[common_numeric].fillna(0))
                
                # Calculate distances
                distances = euclidean_distances(synth_scaled, orig_scaled)
                min_distances = np.min(distances, axis=1)
                
                results['distance_to_closest_record'] = {
                    'mean_min_distance': float(np.mean(min_distances)),
                    'median_min_distance': float(np.median(min_distances)),
                    'std_min_distance': float(np.std(min_distances)),
                    'privacy_score': float(np.mean(min_distances))  # Higher is better for privacy
                }
                
        except Exception as e:
            logger.warning(f"Could not calculate distance metrics: {e}")
            results['distance_to_closest_record'] = {'error': str(e)}
        
        # Membership inference protection
        results['membership_inference'] = {
            'evaluated': False,
            'note': 'Advanced membership inference testing requires specialized tools'
        }
        
        # Attribute disclosure risk
        results['attribute_disclosure'] = {
            'unique_combinations_original': int(self.original_data.nunique().prod()) if len(self.original_data.columns) < 10 else 'Too many to calculate',
            'unique_combinations_synthetic': int(self.synthetic_data.nunique().prod()) if len(self.synthetic_data.columns) < 10 else 'Too many to calculate',
            'note': 'Lower unique combinations in synthetic data indicate better privacy protection'
        }
        
        self.evaluation_results['privacy_protection'] = results
        logger.info("Privacy protection evaluation completed")
    
    def generate_visualizations(self):
        """Generate evaluation visualizations."""
        logger = logging.getLogger(__name__)
        logger.info("Generating evaluation visualizations...")
        
        # Create output directory
        viz_dir = LOGS_DIR / 'evaluation_plots'
        viz_dir.mkdir(exist_ok=True)
        
        plt.style.use('default')
        
        # 1. Age distribution comparison
        if 'age' in self.original_data.columns and 'age' in self.synthetic_data.columns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.original_data['age'], bins=20, alpha=0.7, label='Original', density=True)
            plt.hist(self.synthetic_data['age'], bins=20, alpha=0.7, label='Synthetic', density=True)
            plt.xlabel('Age')
            plt.ylabel('Density')
            plt.title('Age Distribution Comparison')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.boxplot([self.original_data['age'], self.synthetic_data['age']], 
                       labels=['Original', 'Synthetic'])
            plt.ylabel('Age')
            plt.title('Age Distribution Box Plot')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'age_distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Length of stay comparison
        if 'hospital_los_days' in self.original_data.columns and 'hospital_los_days' in self.synthetic_data.columns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.original_data['hospital_los_days'], bins=20, alpha=0.7, label='Original', density=True)
            plt.hist(self.synthetic_data['hospital_los_days'], bins=20, alpha=0.7, label='Synthetic', density=True)
            plt.xlabel('Hospital LOS (days)')
            plt.ylabel('Density')
            plt.title('Hospital LOS Distribution')
            plt.legend()
            
            if 'icu_los_days' in self.original_data.columns and 'icu_los_days' in self.synthetic_data.columns:
                plt.subplot(1, 2, 2)
                plt.hist(self.original_data['icu_los_days'], bins=20, alpha=0.7, label='Original', density=True)
                plt.hist(self.synthetic_data['icu_los_days'], bins=20, alpha=0.7, label='Synthetic', density=True)
                plt.xlabel('ICU LOS (days)')
                plt.ylabel('Density')
                plt.title('ICU LOS Distribution')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'los_distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Correlation matrix comparison
        numeric_cols = ['age', 'hospital_los_days', 'icu_los_days', 'comorbidity_score', 'num_diagnoses']
        available_cols = [col for col in numeric_cols if col in self.original_data.columns and col in self.synthetic_data.columns]
        
        if len(available_cols) > 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original correlation
            orig_corr = self.original_data[available_cols].corr()
            sns.heatmap(orig_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0])
            axes[0].set_title('Original Data Correlations')
            
            # Synthetic correlation
            synth_corr = self.synthetic_data[available_cols].corr()
            sns.heatmap(synth_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1])
            axes[1].set_title('Synthetic Data Correlations')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'correlation_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Primary diagnosis distribution comparison
        if 'primary_diagnosis' in self.original_data.columns and 'primary_diagnosis' in self.synthetic_data.columns:
            orig_counts = self.original_data['primary_diagnosis'].value_counts(normalize=True).head(10)
            synth_counts = self.synthetic_data['primary_diagnosis'].value_counts(normalize=True).head(10)
            
            plt.figure(figsize=(12, 6))
            x = range(len(orig_counts))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], orig_counts.values, width, label='Original', alpha=0.7)
            plt.bar([i + width/2 for i in x], [synth_counts.get(dx, 0) for dx in orig_counts.index], 
                   width, label='Synthetic', alpha=0.7)
            
            plt.xlabel('Diagnosis')
            plt.ylabel('Proportion')
            plt.title('Top 10 Diagnosis Distribution Comparison')
            plt.xticks(x, orig_counts.index, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_dir / 'diagnosis_distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to: {viz_dir}")
        self.evaluation_results['visualizations_path'] = str(viz_dir)
    
    def compile_evaluation_report(self):
        """Compile comprehensive evaluation report."""
        logger = logging.getLogger(__name__)
        logger.info("Compiling evaluation report...")
        
        # Calculate overall scores
        statistical_score = self.evaluation_results.get('statistical_similarity', {}).get('summary_statistics', {}).get('overall_numeric_similarity', 0)
        clinical_score = self._calculate_clinical_score()
        privacy_score = self.evaluation_results.get('privacy_protection', {}).get('distance_to_closest_record', {}).get('privacy_score', 0)
        
        # Normalize privacy score (assuming 0-10 range, higher is better)
        privacy_score_normalized = min(privacy_score / 10, 1.0) if privacy_score > 0 else 0.5
        
        overall_score = (statistical_score + clinical_score + privacy_score_normalized) / 3
        
        # Determine quality rating
        if overall_score >= 0.8:
            quality_rating = "Excellent"
        elif overall_score >= 0.6:
            quality_rating = "Good"
        elif overall_score >= 0.4:
            quality_rating = "Fair"
        else:
            quality_rating = "Needs Improvement"
        
        report_summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'original_samples': len(self.original_data),
                'synthetic_samples': len(self.synthetic_data),
                'model_type': 'CTGAN'
            },
            'overall_evaluation': {
                'overall_score': round(overall_score, 3),
                'quality_rating': quality_rating,
                'statistical_similarity_score': round(statistical_score, 3),
                'clinical_validity_score': round(clinical_score, 3),
                'privacy_protection_score': round(privacy_score_normalized, 3)
            },
            'detailed_results': self.evaluation_results,
            'recommendations': self._generate_recommendations(overall_score, statistical_score, clinical_score, privacy_score_normalized)
        }
        
        # Save detailed report
        report_path = MODELS_DIR / 'model_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_summary, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        return report_summary, report_path
    
    def _calculate_clinical_score(self):
        """Calculate overall clinical validity score."""
        clinical_results = self.evaluation_results.get('clinical_validity', {})
        
        scores = []
        
        # Age distribution score
        age_diff = clinical_results.get('age_distribution', {}).get('age_difference', 10)
        age_score = max(0, 1 - age_diff / 10)  # Penalize large age differences
        scores.append(age_score)
        
        # Clinical logic score
        clinical_logic = clinical_results.get('clinical_logic', {})
        logic_checks = [
            clinical_logic.get('icu_los_consistency', {}).get('icu_patients_have_los', True),
            clinical_logic.get('icu_los_consistency', {}).get('non_icu_patients_zero_los', True),
            clinical_logic.get('hospital_icu_los_relationship', True)
        ]
        logic_score = sum(logic_checks) / len(logic_checks) if logic_checks else 0.5
        scores.append(logic_score)
        
        # Mortality pattern score
        mortality_diff = clinical_results.get('mortality_patterns', {}).get('mortality_rate_difference', 0.5)
        mortality_score = max(0, 1 - mortality_diff * 5)  # Penalize large mortality differences
        scores.append(mortality_score)
        
        # ICU pattern score
        icu_diff = clinical_results.get('icu_patterns', {}).get('icu_rate_difference', 0.5)
        icu_score = max(0, 1 - icu_diff * 3)  # Penalize large ICU rate differences
        scores.append(icu_score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _generate_recommendations(self, overall_score, statistical_score, clinical_score, privacy_score):
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if overall_score < 0.6:
            recommendations.append("Overall model performance needs improvement. Consider retraining with different parameters.")
        
        if statistical_score < 0.7:
            recommendations.append("Statistical similarity is low. Consider:")
            recommendations.append("- Increasing training epochs")
            recommendations.append("- Adjusting generator/discriminator architecture")
            recommendations.append("- Reviewing data preprocessing steps")
        
        if clinical_score < 0.7:
            recommendations.append("Clinical validity needs attention. Consider:")
            recommendations.append("- Adding post-processing constraints")
            recommendations.append("- Implementing clinical logic validation")
            recommendations.append("- Reviewing feature engineering for clinical relationships")
        
        if privacy_score < 0.5:
            recommendations.append("Privacy protection may be insufficient. Consider:")
            recommendations.append("- Adding differential privacy mechanisms")
            recommendations.append("- Implementing additional noise injection")
            recommendations.append("- Reviewing distance-based privacy metrics")
        
        if statistical_score > 0.8 and clinical_score > 0.8:
            recommendations.append("Excellent model performance! The synthetic data maintains high quality and clinical validity.")
        
        return recommendations

def main():
    """Main evaluation execution."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting CTGAN model evaluation...")
        
        evaluator = CTGANModelEvaluator()
        
        # Load model and data
        print("\n" + "="*60)
        print("CTGAN MODEL EVALUATION")
        print("="*60)
        print("Loading model and data...")
        evaluator.load_model_and_data()
        
        # Generate synthetic data for evaluation
        print("Generating synthetic data for evaluation...")
        evaluator.generate_synthetic_data()
        
        # Run evaluations
        print("Running statistical similarity evaluation...")
        evaluator.evaluate_statistical_similarity()
        
        print("Running clinical validity evaluation...")
        evaluator.evaluate_clinical_validity()
        
        print("Running utility preservation evaluation...")
        evaluator.evaluate_utility_preservation()
        
        print("Running privacy protection evaluation...")
        evaluator.evaluate_privacy_protection()
        
        print("Generating visualizations...")
        evaluator.generate_visualizations()
        
        # Compile report
        print("Compiling evaluation report...")
        report_summary, report_path = evaluator.compile_evaluation_report()
        
        # Print summary results
        print("\n" + "="*60)
        print("EVALUATION RESULTS SUMMARY")
        print("="*60)
        
        overall_eval = report_summary['overall_evaluation']
        print(f"Overall Quality Rating: {overall_eval['quality_rating']}")
        print(f"Overall Score: {overall_eval['overall_score']:.3f}")
        
        print(f"\nDetailed Scores:")
        print(f"  Statistical Similarity: {overall_eval['statistical_similarity_score']:.3f}")
        print(f"  Clinical Validity: {overall_eval['clinical_validity_score']:.3f}")
        print(f"  Privacy Protection: {overall_eval['privacy_protection_score']:.3f}")
        
        # Statistical similarity details
        stats_results = report_summary['detailed_results'].get('statistical_similarity', {})
        if 'summary_statistics' in stats_results:
            summary_stats = stats_results['summary_statistics']
            print(f"\nStatistical Details:")
            print(f"  Numeric similarity: {summary_stats.get('overall_numeric_similarity', 0):.3f}")
            print(f"  Categorical similarity: {summary_stats.get('overall_categorical_similarity', 0):.3f}")
            print(f"  Columns evaluated: {summary_stats.get('total_columns_evaluated', 0)}")
        
        # Clinical validity details
        clinical_results = report_summary['detailed_results'].get('clinical_validity', {})
        if 'age_distribution' in clinical_results:
            age_dist = clinical_results['age_distribution']
            print(f"\nClinical Validity Details:")
            print(f"  Age difference: {age_dist.get('age_difference', 0):.1f} years")
            print(f"  Adults only (synthetic): {age_dist.get('adults_only_synthetic', 'Unknown')}")
        
        if 'mortality_patterns' in clinical_results:
            mortality = clinical_results['mortality_patterns']
            print(f"  Mortality rate (original): {mortality.get('original_mortality_rate', 0):.3f}")
            print(f"  Mortality rate (synthetic): {mortality.get('synthetic_mortality_rate', 0):.3f}")
        
        if 'icu_patterns' in clinical_results:
            icu = clinical_results['icu_patterns']
            print(f"  ICU rate (original): {icu.get('original_icu_rate', 0):.3f}")
            print(f"  ICU rate (synthetic): {icu.get('synthetic_icu_rate', 0):.3f}")
        
        # Utility preservation
        utility_results = report_summary['detailed_results'].get('utility_preservation', {})
        if 'ml_utility' in utility_results and 'utility_preservation_score' in utility_results['ml_utility']:
            ml_utility = utility_results['ml_utility']
            print(f"\nUtility Preservation:")
            print(f"  ML utility score: {ml_utility.get('utility_preservation_score', 0):.3f}")
            print(f"  Accuracy ratio: {ml_utility.get('accuracy_ratio', 0):.3f}")
        
        # Privacy protection
        privacy_results = report_summary['detailed_results'].get('privacy_protection', {})
        if 'distance_to_closest_record' in privacy_results:
            distance_metrics = privacy_results['distance_to_closest_record']
            if 'mean_min_distance' in distance_metrics:
                print(f"\nPrivacy Protection:")
                print(f"  Mean distance to closest record: {distance_metrics.get('mean_min_distance', 0):.3f}")
        
        # Recommendations
        recommendations = report_summary.get('recommendations', [])
        if recommendations:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"  {i}. {rec}")
        
        # Output files
        viz_path = report_summary['detailed_results'].get('visualizations_path', 'Not generated')
        print(f"\nOutput Files:")
        print(f"  Evaluation report: {report_path}")
        print(f"  Visualizations: {viz_path}")
        
        # Overall assessment
        if overall_eval['overall_score'] >= 0.8:
            print(f"\n🎉 Excellent! The model generates high-quality synthetic data.")
        elif overall_eval['overall_score'] >= 0.6:
            print(f"\n✅ Good performance. The model is suitable for most use cases.")
        elif overall_eval['overall_score'] >= 0.4:
            print(f"\n⚠️  Fair performance. Consider improvements before production use.")
        else:
            print(f"\n❌ Performance needs significant improvement before use.")
        
        logger.info("Model evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check the logs for more details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)