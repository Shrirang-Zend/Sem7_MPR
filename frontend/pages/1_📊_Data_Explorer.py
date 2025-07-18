#!/usr/bin/env python3
"""
Data Explorer Page - Interactive data exploration and visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add backend path for importing config
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Add frontend path for importing components
frontend_path = Path(__file__).parent.parent
sys.path.insert(0, str(frontend_path))

from components.api_client import APIClient
from utils.data_processing import process_generated_data, create_visualizations
from backend.config.settings import API_CONFIG

st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Initialize API client
API_BASE_URL = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}"
api_client = APIClient(API_BASE_URL)

st.title("📊 Healthcare Data Explorer")
st.markdown("Explore and analyze healthcare datasets with interactive visualizations")

# Sidebar controls
with st.sidebar:
    st.header("🔧 Explorer Controls")
    
    # Data source selection
    data_source = st.selectbox(
        "Data Source:",
        ["Generated Data", "Upload CSV", "Sample Dataset"]
    )
    
    if data_source == "Generated Data":
        st.info("Using data from the main generation interface")
        df = None
        if 'generated_data' in st.session_state and st.session_state.generated_data:
            data = st.session_state.generated_data
            if data.get("success") and "data" in data:
                df = process_generated_data(data["data"])
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        df = None
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows")
    
    else:  # Sample Dataset
        df = None
        if st.button("Load Sample Data"):
            # Generate sample data via API
            sample_data = api_client.generate_data(
                "Generate 50 diverse patients with various conditions",
                num_patients=50,
                format_type="json"
            )
            if sample_data.get("success"):
                df = process_generated_data(sample_data["data"])
                st.success("Sample data loaded!")

# Main content
if df is not None and not df.empty:
    # Dataset overview
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Interactive analysis
    st.header("🔬 Interactive Analysis")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "Column Analysis", "Correlations", "Custom Plots"
    ])
    
    with analysis_tab1:
        st.subheader("📊 Column Analysis")
        
        selected_column = st.selectbox(
            "Select column to analyze:",
            df.columns
        )
        
        col_data = df[selected_column]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Statistics:**")
            if col_data.dtype in ['float64', 'int64']:
                stats = col_data.describe()
                for stat, value in stats.items():
                    st.write(f"- {stat.title()}: {value:.2f}")
            else:
                st.write(f"- Data Type: {col_data.dtype}")
                st.write(f"- Unique Values: {col_data.nunique()}")
                st.write(f"- Missing Values: {col_data.isnull().sum()}")
        
        with col2:
            st.write("**Visualization:**")
            if col_data.dtype in ['float64', 'int64']:
                fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                value_counts = col_data.value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Top 10 values in {selected_column}")
                st.plotly_chart(fig, use_container_width=True)
    
    with analysis_tab2:
        st.subheader("🔗 Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            st.subheader("🎯 Strong Correlations")
            threshold = st.slider("Correlation threshold:", 0.5, 1.0, 0.7)
            
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        strong_corrs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if strong_corrs:
                corr_df = pd.DataFrame(strong_corrs)
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.info(f"No correlations above {threshold:.1f} threshold")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    with analysis_tab3:
        st.subheader("📈 Custom Plots")
        
        plot_type = st.selectbox(
            "Plot Type:",
            ["Scatter Plot", "Box Plot", "Violin Plot", "Histogram", "Bar Chart"]
        )
        
        if plot_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis:", df.columns, key="scatter_x")
            with col2:
                y_axis = st.selectbox("Y-axis:", df.columns, key="scatter_y")
            
            color_by = st.selectbox("Color by (optional):", [None] + list(df.columns))
            
            if x_axis != y_axis:
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, 
                    color=color_by if color_by else None,
                    title=f"{y_axis} vs {x_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            col1, col2 = st.columns(2)
            with col1:
                y_axis = st.selectbox("Value column:", df.select_dtypes(include=['float64', 'int64']).columns)
            with col2:
                x_axis = st.selectbox("Group by:", [None] + list(df.columns))
            
            fig = px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {y_axis}")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please select a data source from the sidebar to begin exploration")
    
    # Show available options
    st.markdown("""
    ## 📚 Available Data Sources:
    
    1. **Generated Data** - Use data from the main API testing interface
    2. **Upload CSV** - Upload your own healthcare dataset
    3. **Sample Dataset** - Generate sample data via the API
    
    ## 🔍 Features Available:
    
    - Dataset overview and statistics
    - Interactive column analysis
    - Correlation analysis with heatmaps
    - Custom visualizations (scatter, box, violin plots)
    - Data quality metrics
    - Export capabilities
    """)

# Footer
st.markdown("---")
st.markdown("*Healthcare Data Explorer - Part of the Healthcare Data Generation System*")