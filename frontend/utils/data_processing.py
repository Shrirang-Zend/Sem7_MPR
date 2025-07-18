"""
Data Processing Utilities for Healthcare Data Generation Frontend
Functions for processing and analyzing generated healthcare data
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Any, Optional


def process_generated_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process raw generated data into a pandas DataFrame
    
    Args:
        data: List of dictionaries containing patient data
    
    Returns:
        Processed pandas DataFrame
    """
    
    if not data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Clean and standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Process common healthcare data types
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    if 'patient_id' in df.columns:
        df['patient_id'] = df['patient_id'].astype(str)
    
    # Process gender column
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.upper()
    
    # Process boolean columns
    boolean_columns = ['has_icu_stay', 'mortality', 'emergency_admission', 'high_complexity']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Process numeric columns
    numeric_columns = ['hospital_los_days', 'icu_los_days', 'total_cost']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def create_visualizations(df: pd.DataFrame):
    """
    Create visualizations for the healthcare data
    
    Args:
        df: Processed pandas DataFrame
    """
    
    if df.empty:
        st.warning("No data available for visualization")
        return
    
    st.subheader("📊 Data Visualizations")
    
    # Create tabs for different visualization categories
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Demographics", "Clinical", "Outcomes"])
    
    with viz_tab1:
        render_demographic_charts(df)
    
    with viz_tab2:
        render_clinical_charts(df)
    
    with viz_tab3:
        render_outcome_charts(df)


def render_demographic_charts(df: pd.DataFrame):
    """Render demographic visualization charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        if 'age' in df.columns and df['age'].notna().any():
            st.subheader("👥 Age Distribution")
            
            fig_age = px.histogram(
                df, 
                x='age', 
                nbins=20,
                title="Age Distribution",
                labels={'age': 'Age (years)', 'count': 'Number of Patients'}
            )
            fig_age.update_layout(showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Age statistics
            age_stats = df['age'].describe()
            st.write("**Age Statistics:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Mean", f"{age_stats['mean']:.1f}")
            with col_b:
                st.metric("Median", f"{age_stats['50%']:.1f}")
            with col_c:
                st.metric("Std Dev", f"{age_stats['std']:.1f}")
        else:
            st.info("No age data available")
    
    with col2:
        # Gender distribution
        if 'gender' in df.columns and df['gender'].notna().any():
            st.subheader("⚥ Gender Distribution")
            
            gender_counts = df['gender'].value_counts()
            
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution"
            )
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Gender statistics
            st.write("**Gender Breakdown:**")
            for gender, count in gender_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"- {gender}: {count} ({percentage:.1f}%)")
        else:
            st.info("No gender data available")


def render_clinical_charts(df: pd.DataFrame):
    """Render clinical visualization charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ICU stay analysis
        icu_col = find_column(df, ['has_icu_stay', 'icu_stay', 'icu'])
        if icu_col:
            st.subheader("🏥 ICU Stay Analysis")
            
            icu_counts = df[icu_col].value_counts()
            
            # Handle different data types
            if df[icu_col].dtype == 'bool':
                labels = ["No ICU Stay", "ICU Stay"]
                values = [icu_counts.get(False, 0), icu_counts.get(True, 0)]
            else:
                labels = [str(label) for label in icu_counts.index]
                values = icu_counts.values
            
            fig_icu = px.pie(
                values=values,
                names=labels,
                title="ICU Stay Distribution",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig_icu, use_container_width=True)
        else:
            st.info("No ICU data available")
        
        # Hospital length of stay
        los_col = find_column(df, ['hospital_los_days', 'los_days', 'length_of_stay'])
        if los_col and df[los_col].notna().any():
            st.subheader("📅 Length of Stay")
            
            fig_los = px.histogram(
                df,
                x=los_col,
                nbins=20,
                title="Hospital Length of Stay Distribution",
                labels={los_col: 'Days', 'count': 'Number of Patients'}
            )
            st.plotly_chart(fig_los, use_container_width=True)
            
            # LOS statistics
            los_stats = df[los_col].describe()
            st.write("**LOS Statistics:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Mean", f"{los_stats['mean']:.1f} days")
            with col_b:
                st.metric("Median", f"{los_stats['50%']:.1f} days")
            with col_c:
                st.metric("Max", f"{los_stats['max']:.0f} days")
    
    with col2:
        # Diagnosis analysis
        diag_col = find_column(df, ['diagnoses', 'diagnosis', 'primary_diagnosis'])
        if diag_col and df[diag_col].notna().any():
            st.subheader("🔬 Diagnosis Analysis")
            
            # Process diagnosis data (handle multiple diagnoses)
            all_diagnoses = []
            for diagnoses in df[diag_col].dropna():
                if isinstance(diagnoses, str):
                    # Split multiple diagnoses if separated by commas
                    diag_list = [d.strip() for d in diagnoses.split(',')]
                    all_diagnoses.extend(diag_list)
                else:
                    all_diagnoses.append(str(diagnoses))
            
            if all_diagnoses:
                diag_series = pd.Series(all_diagnoses)
                top_diagnoses = diag_series.value_counts().head(10)
                
                fig_diag = px.bar(
                    x=top_diagnoses.values,
                    y=top_diagnoses.index,
                    orientation='h',
                    title="Top 10 Diagnoses",
                    labels={'x': 'Count', 'y': 'Diagnosis'}
                )
                fig_diag.update_layout(height=400)
                st.plotly_chart(fig_diag, use_container_width=True)
            else:
                st.info("No diagnosis data to analyze")
        else:
            st.info("No diagnosis data available")


def render_outcome_charts(df: pd.DataFrame):
    """Render outcome visualization charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mortality analysis
        mortality_col = find_column(df, ['mortality', 'death', 'died', 'deceased'])
        if mortality_col:
            st.subheader("💔 Mortality Analysis")
            
            mortality_counts = df[mortality_col].value_counts()
            
            # Handle different data types
            if df[mortality_col].dtype == 'bool':
                labels = ["Survived", "Mortality"]
                values = [mortality_counts.get(False, 0), mortality_counts.get(True, 0)]
                colors = ['#2ECC71', '#E74C3C']
            else:
                labels = [str(label) for label in mortality_counts.index]
                values = mortality_counts.values
                colors = ['#2ECC71', '#E74C3C']
            
            fig_mortality = px.pie(
                values=values,
                names=labels,
                title="Mortality Outcomes",
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig_mortality, use_container_width=True)
            
            # Mortality rate
            if df[mortality_col].dtype == 'bool':
                mortality_rate = df[mortality_col].mean() * 100
                st.metric("Mortality Rate", f"{mortality_rate:.1f}%")
        else:
            st.info("No mortality data available")
    
    with col2:
        # Emergency admission analysis
        emergency_col = find_column(df, ['emergency_admission', 'emergency', 'admit_type'])
        if emergency_col:
            st.subheader("🚨 Emergency Admissions")
            
            emergency_counts = df[emergency_col].value_counts()
            
            fig_emergency = px.bar(
                x=emergency_counts.index,
                y=emergency_counts.values,
                title="Emergency vs Regular Admissions",
                labels={'x': 'Admission Type', 'y': 'Count'}
            )
            st.plotly_chart(fig_emergency, use_container_width=True)
        else:
            st.info("No emergency admission data available")
    
    # Correlation analysis if multiple numeric columns exist
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find a column in the DataFrame that matches one of the possible names
    
    Args:
        df: DataFrame to search
        possible_names: List of possible column names
    
    Returns:
        Column name if found, None otherwise
    """
    
    df_columns_lower = [col.lower() for col in df.columns]
    
    for name in possible_names:
        if name.lower() in df_columns_lower:
            # Return the actual column name (with original case)
            idx = df_columns_lower.index(name.lower())
            return df.columns[idx]
    
    return None


def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality metrics for the DataFrame
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary containing quality metrics
    """
    
    if df.empty:
        return {"error": "No data to analyze"}
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = 1 - (missing_cells / total_cells)
    
    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    uniqueness = 1 - (duplicate_rows / len(df))
    
    # Data type consistency
    consistent_types = 0
    total_columns = len(df.columns)
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'bool', 'datetime64[ns]']:
            consistent_types += 1
    
    consistency = consistent_types / total_columns if total_columns > 0 else 0
    
    # Overall quality score
    quality_score = (completeness + uniqueness + consistency) / 3
    
    return {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "consistency": consistency,
        "quality_score": quality_score,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "total_rows": len(df),
        "total_columns": len(df.columns)
    }


def export_data_summary(df: pd.DataFrame) -> str:
    """
    Generate a text summary of the DataFrame
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        Text summary string
    """
    
    if df.empty:
        return "No data available"
    
    summary = f"""
    Healthcare Data Summary
    ======================
    
    Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns
    
    Columns:
    {', '.join(df.columns)}
    
    Data Types:
    {df.dtypes.to_string()}
    
    Missing Values:
    {df.isnull().sum().to_string()}
    
    Numeric Summary:
    {df.describe().to_string()}
    """
    
    return summary