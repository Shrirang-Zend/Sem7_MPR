#!/usr/bin/env python3
"""
Model Monitor Page - Monitor CTGAN model performance and training metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add backend path for importing config
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Add frontend path for importing components
frontend_path = Path(__file__).parent.parent
sys.path.insert(0, str(frontend_path))

# Define project root path
project_root = Path(__file__).parent.parent.parent

from components.api_client import APIClient
from config.settings import API_CONFIG

st.set_page_config(
    page_title="Model Monitor",
    page_icon="🔧",
    layout="wide"
)

# Initialize API client
API_BASE_URL = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}"
api_client = APIClient(API_BASE_URL)

st.title("🔧 CTGAN Model Monitor")
st.markdown("Monitor model performance, training metrics, and system health")

# Auto-refresh option
with st.sidebar:
    st.header("⚙️ Monitor Settings")
    
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_interval = st.slider("Refresh interval (seconds):", 5, 60, 10)
    
    if auto_refresh:
        st.info(f"Auto-refreshing every {refresh_interval} seconds")
        time.sleep(refresh_interval)
        st.rerun()
    
    if st.button("🔄 Manual Refresh"):
        st.rerun()
    
    st.markdown("---")
    
    # Model actions
    st.header("🎛️ Model Actions")
    
    if st.button("📊 Get Model Stats", use_container_width=True):
        st.session_state.show_model_stats = True
    
    if st.button("🔍 Check Model Health", use_container_width=True):
        st.session_state.show_health_check = True
    
    if st.button("📈 Performance Test", use_container_width=True):
        st.session_state.show_performance_test = True

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 System Health", "📊 Model Statistics", "📈 Performance", "🔧 Diagnostics"
])

with tab1:
    st.header("🏥 System Health Overview")
    
    # Health check
    health_status = api_client.health_check()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if health_status.get("status") == "healthy":
            st.success("✅ API Service: Healthy")
        else:
            st.error("❌ API Service: Unhealthy")
            st.write(f"Error: {health_status.get('message', 'Unknown')}")
    
    with col2:
        # Test model availability
        stats_response = api_client.get_statistics()
        if "error" not in stats_response:
            st.success("✅ Model: Available")
        else:
            st.error("❌ Model: Unavailable")
    
    with col3:
        # Response time test
        start_time = time.time()
        api_client.health_check()
        response_time = (time.time() - start_time) * 1000
        
        if response_time < 100:
            st.success(f"✅ Response: {response_time:.0f}ms")
        elif response_time < 500:
            st.warning(f"⚠️ Response: {response_time:.0f}ms")
        else:
            st.error(f"❌ Response: {response_time:.0f}ms")
    
    # System metrics
    if health_status.get("status") == "healthy":
        st.subheader("🖥️ System Metrics")
        
        # Check if health endpoint provides system info
        if "system_info" in health_status:
            system_info = health_status["system_info"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_usage = system_info.get("cpu_usage", 0)
                st.metric("CPU Usage", f"{cpu_usage:.1f}%")
            
            with col2:
                memory_usage = system_info.get("memory_usage", 0)
                st.metric("Memory Usage", f"{memory_usage:.1f}%")
            
            with col3:
                disk_usage = system_info.get("disk_usage", 0)
                st.metric("Disk Usage", f"{disk_usage:.1f}%")
            
            with col4:
                uptime = system_info.get("uptime", "Unknown")
                st.metric("Uptime", uptime)
        else:
            st.info("Detailed system metrics not available from API")

with tab2:
    st.header("📊 Model Statistics & Data Quality")
    
    stats = api_client.get_statistics()
    
    if "error" not in stats:
        # Core statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_patients = stats.get("total_patients", 0)
            st.metric("Training Patients", f"{total_patients:,}")
        
        with col2:
            avg_age = stats.get("avg_age", 0)
            st.metric("Average Age", f"{avg_age:.1f}")
        
        with col3:
            unique_diagnoses = stats.get("unique_diagnoses", 0)
            st.metric("Unique Diagnoses", unique_diagnoses)
        
        with col4:
            quality_score = stats.get("quality_score", 0)
            st.metric("Data Quality", f"{quality_score:.1%}")
        
        # Model performance metrics
        st.subheader("🎯 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_accuracy = stats.get("model_accuracy", 0)
            st.metric("Model Accuracy", f"{model_accuracy:.1%}")
        
        with col2:
            icu_rate = stats.get("icu_rate", 0)
            st.metric("ICU Rate", f"{icu_rate:.1%}")
        
        with col3:
            mortality_rate = stats.get("mortality_rate", 0)
            st.metric("Mortality Rate", f"{mortality_rate:.1%}")
        
        # Data distributions
        if "diagnosis_distribution" in stats:
            st.subheader("📈 Training Data Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Diagnosis distribution
                diag_data = stats["diagnosis_distribution"]
                if diag_data and len(diag_data) > 0:
                    df_diag = pd.DataFrame(
                        list(diag_data.items()), 
                        columns=["Diagnosis", "Count"]
                    ).head(10)
                    
                    fig = px.bar(
                        df_diag, 
                        x="Count", 
                        y="Diagnosis", 
                        orientation="h",
                        title="Top 10 Diagnoses in Training Data"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age distribution
                if "age_distribution" in stats:
                    age_data = stats["age_distribution"]
                    if age_data:
                        ages = [int(k) for k in age_data.keys()]
                        counts = list(age_data.values())
                        
                        fig = px.histogram(
                            x=ages, 
                            y=counts,
                            title="Age Distribution in Training Data"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"❌ Error fetching statistics: {stats.get('error')}")

with tab3:
    st.header("📈 Performance Testing")
    
    # Performance test controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.selectbox("Test Data Size:", [10, 50, 100, 500])
    
    with col2:
        num_tests = st.selectbox("Number of Tests:", [1, 5, 10])
    
    with col3:
        if st.button("🚀 Run Performance Test"):
            st.session_state.run_performance_test = True
    
    # Run performance test
    if st.session_state.get("run_performance_test", False):
        st.subheader("⏱️ Performance Test Results")
        
        progress_bar = st.progress(0)
        results = []
        
        for i in range(num_tests):
            start_time = time.time()
            
            # Generate test data
            test_result = api_client.generate_data(
                f"Generate {test_size} test patients",
                num_patients=test_size,
                format_type="json"
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            results.append({
                "Test": i + 1,
                "Patients": test_size,
                "Response Time (ms)": response_time,
                "Success": test_result.get("success", False)
            })
            
            progress_bar.progress((i + 1) / num_tests)
        
        # Display results
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_response = df_results["Response Time (ms)"].mean()
            st.metric("Avg Response Time", f"{avg_response:.0f}ms")
        
        with col2:
            min_response = df_results["Response Time (ms)"].min()
            st.metric("Min Response Time", f"{min_response:.0f}ms")
        
        with col3:
            max_response = df_results["Response Time (ms)"].max()
            st.metric("Max Response Time", f"{max_response:.0f}ms")
        
        with col4:
            success_rate = df_results["Success"].mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Response time chart
        fig = px.line(
            df_results, 
            x="Test", 
            y="Response Time (ms)",
            title="Response Time by Test",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.run_performance_test = False

with tab4:
    st.header("🔧 System Diagnostics")
    
    # API endpoint tests
    st.subheader("🌐 API Endpoint Tests")
    
    endpoints = [
        {"path": "/health", "method": "GET", "description": "Health check"},
        {"path": "/statistics", "method": "GET", "description": "Get statistics"},
        {"path": "/generate", "method": "POST", "description": "Generate data"}
    ]
    
    for endpoint in endpoints:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**{endpoint['method']} {endpoint['path']}**")
        
        with col2:
            st.write(endpoint['description'])
        
        with col3:
            if st.button(f"Test", key=f"test_{endpoint['path']}"):
                if endpoint['method'] == 'GET':
                    result = api_client.test_endpoint(endpoint['path'])
                else:
                    result = api_client.test_endpoint(
                        endpoint['path'], 
                        method='POST',
                        data={"query": "test", "num_patients": 1, "format": "json"}
                    )
                
                st.session_state[f"test_result_{endpoint['path']}"] = result
        
        with col4:
            result = st.session_state.get(f"test_result_{endpoint['path']}")
            if result:
                if result.get("success"):
                    st.success(f"✅ {result.get('response_time_ms', 0):.0f}ms")
                else:
                    st.error("❌ Failed")
    
    # Configuration check
    st.subheader("⚙️ Configuration Check")
    
    config_items = [
        ("API Host", API_CONFIG.get('host', 'Unknown')),
        ("API Port", API_CONFIG.get('port', 'Unknown')),
        ("Max Patients", API_CONFIG.get('max_generated_rows', 'Unknown')),
        ("Timeout", f"{API_CONFIG.get('timeout_seconds', 'Unknown')}s")
    ]
    
    for item, value in config_items:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{item}:**")
        with col2:
            st.write(str(value))
    
    # Log viewer (if available)
    st.subheader("📋 Recent Activity")
    
    # Try to read recent logs
    log_path = project_root / "logs" / "healthcare_system.log"
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                recent_logs = lines[-20:]  # Last 20 lines
            
            st.text_area("Recent Log Entries", "".join(recent_logs), height=200)
        except Exception as e:
            st.error(f"Error reading logs: {str(e)}")
    else:
        st.info("Log file not found")

# Footer
st.markdown("---")
st.markdown("*Model Monitor - Real-time monitoring for Healthcare Data Generation System*")