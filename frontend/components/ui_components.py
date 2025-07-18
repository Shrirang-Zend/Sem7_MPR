"""
UI Components for Healthcare Data Generation Frontend
Reusable Streamlit components for consistent styling and functionality
"""

import streamlit as st
from typing import Dict, Any


def render_header():
    """Render the main application header with styling"""
    
    # Custom CSS for header styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .header-title {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header-subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.2rem;
            margin: 0.5rem 0 0 0;
        }
        .status-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-container {
            text-align: center;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header content
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">🏥 Healthcare Data System</h1>
        <p class="header-subtitle">Synthetic Medical Data Generation & Testing Platform</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(api_client, health_status: Dict[str, Any]):
    """
    Render the sidebar with system status and controls
    
    Args:
        api_client: API client instance
        health_status: Current health status from API
    """
    
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Refresh button
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        # Connection status
        if health_status.get("status") == "healthy":
            st.success("✅ API Connected")
            
            # Try to get additional system info
            try:
                stats = api_client.get_statistics()
                if "error" not in stats:
                    st.info("📊 Data Available")
                    
                    # Display key metrics in compact format
                    col1, col2 = st.columns(2)
                    with col1:
                        total_patients = stats.get("total_patients", 0)
                        st.metric("Patients", f"{total_patients:,}")
                    
                    with col2:
                        avg_age = stats.get("avg_age", 0)
                        st.metric("Avg Age", f"{avg_age:.0f}")
                else:
                    st.warning("⚠️ Limited Data")
            
            except Exception:
                st.warning("⚠️ Stats Unavailable")
        
        else:
            st.error("❌ API Disconnected")
            st.write("**Issue:**")
            st.write(health_status.get("message", "Unknown error"))
        
        st.markdown("---")
        
        # Settings section
        st.header("⚙️ Settings")
        
        # Default parameters
        default_patients = st.number_input(
            "Default Patient Count:",
            min_value=1,
            max_value=1000,
            value=10,
            help="Default number of patients for generation"
        )
        
        default_format = st.selectbox(
            "Default Format:",
            ["json", "csv"],
            help="Default output format for generated data"
        )
        
        # Store in session state
        st.session_state.default_patients = default_patients
        st.session_state.default_format = default_format
        
        # API endpoint info
        st.markdown("---")
        st.header("🔗 API Info")
        
        api_info = api_client.get_api_info()
        st.write(f"**Base URL:** `{api_info['base_url']}`")
        st.write(f"**Version:** {api_info['version']}")
        
        # Show available endpoints
        with st.expander("📋 Available Endpoints"):
            for endpoint in api_info['endpoints']:
                st.write(f"**{endpoint['method']}** `{endpoint['path']}`")
                st.write(f"_{endpoint['description']}_")
                st.write("")


def render_status_cards(metadata: Dict[str, Any]):
    """
    Render status cards with key metrics
    
    Args:
        metadata: Metadata dictionary from generated data
    """
    
    st.subheader("📊 Generation Summary")
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = metadata.get("total_patients", 0)
        st.metric(
            "Total Patients",
            f"{total_patients:,}",
            help="Number of synthetic patients generated"
        )
    
    with col2:
        avg_age = metadata.get("avg_age", 0)
        age_std = metadata.get("age_std", 0)
        st.metric(
            "Average Age",
            f"{avg_age:.1f}",
            delta=f"±{age_std:.1f}",
            help="Mean age with standard deviation"
        )
    
    with col3:
        gender_ratio = metadata.get("gender_ratio", {})
        if gender_ratio:
            female_pct = gender_ratio.get("F", 0) * 100
            st.metric(
                "Female %",
                f"{female_pct:.1f}%",
                help="Percentage of female patients"
            )
        else:
            st.metric("Gender", "N/A")
    
    with col4:
        quality_score = metadata.get("quality_score", 0)
        st.metric(
            "Quality Score",
            f"{quality_score:.1%}",
            help="Data quality assessment score"
        )
    
    # Additional metrics if available
    if any(key in metadata for key in ["icu_rate", "mortality_rate", "avg_hospital_los"]):
        st.subheader("🏥 Clinical Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            icu_rate = metadata.get("icu_rate", 0)
            st.metric(
                "ICU Rate",
                f"{icu_rate:.1%}",
                help="Percentage of patients with ICU stay"
            )
        
        with col2:
            mortality_rate = metadata.get("mortality_rate", 0)
            st.metric(
                "Mortality Rate",
                f"{mortality_rate:.1%}",
                help="Percentage of patients with mortality outcome"
            )
        
        with col3:
            avg_los = metadata.get("avg_hospital_los", 0)
            st.metric(
                "Avg LOS",
                f"{avg_los:.1f} days",
                help="Average hospital length of stay"
            )


def render_success_message(message: str, details: str = None):
    """Render a success message with optional details"""
    st.success(f"✅ {message}")
    if details:
        st.info(details)


def render_error_message(message: str, details: str = None):
    """Render an error message with optional details"""
    st.error(f"❌ {message}")
    if details:
        with st.expander("📋 Error Details"):
            st.write(details)


def render_warning_message(message: str, details: str = None):
    """Render a warning message with optional details"""
    st.warning(f"⚠️ {message}")
    if details:
        st.info(details)


def render_api_endpoint_card(endpoint: str, method: str, description: str, 
                            is_available: bool = True):
    """
    Render a card for API endpoint information
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        description: Endpoint description
        is_available: Whether the endpoint is currently available
    """
    
    status_icon = "🟢" if is_available else "🔴"
    status_text = "Available" if is_available else "Unavailable"
    
    st.markdown(f"""
    <div class="status-card">
        <h4>{status_icon} {method} {endpoint}</h4>
        <p><strong>Status:</strong> {status_text}</p>
        <p><strong>Description:</strong> {description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_loading_spinner(message: str = "Processing..."):
    """Render a loading spinner with custom message"""
    return st.spinner(f"⏳ {message}")


def render_data_preview_card(df, max_rows: int = 5):
    """
    Render a preview card for dataframe data
    
    Args:
        df: Pandas DataFrame to preview
        max_rows: Maximum number of rows to show
    """
    
    st.markdown("""
    <div class="status-card">
        <h4>📋 Data Preview</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Size", f"{memory_usage:.1f} MB")
    
    st.dataframe(df.head(max_rows), use_container_width=True)