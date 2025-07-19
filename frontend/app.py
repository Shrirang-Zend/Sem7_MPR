#!/usr/bin/env python3
"""
Streamlit Frontend for Bio.Entrez Enhanced Medical Research API
Interactive web interface for synthetic medical data generation
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Dr. Freak",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://backend:8000"  # Your FastAPI backend
STREAMLIT_PORT = 8501  # Streamlit frontend port

# Initialize session state variables
if 'sample_size_override' not in st.session_state:
    st.session_state.sample_size_override = 100
if 'age_range_override' not in st.session_state:
    st.session_state.age_range_override = None
if 'use_research_context' not in st.session_state:
    st.session_state.use_research_context = True

class APIClient:
    """Client for interacting with the Medical Research API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status_code}"}
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait 1 second before retry
                    continue
                return {"status": "error", "message": "Connection refused - API not responding"}
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {"status": "error", "message": "Request timeout"}
            except Exception as e:
                return {"status": "error", "message": f"Connection error: {str(e)}"}
        
        return {"status": "error", "message": "Max retries exceeded"}
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        try:
            response = self.session.get(f"{self.base_url}/system/status", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_context(self, query: str, max_articles: int = 50) -> Dict:
        """Generate research context"""
        try:
            data = {"query": query, "max_articles": max_articles}
            response = self.session.post(f"{self.base_url}/generate-context", json=data, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False, 
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "query_id": "",
                    "extracted_params": {},
                    "pubmed_results": [],
                    "research_context": ""
                }
        except Exception as e:
            return {
                "success": False, 
                "error": str(e),
                "query_id": "",
                "extracted_params": {},
                "pubmed_results": [],
                "research_context": ""
            }
    
    def parse_query(self, query: str, max_articles: int = 50) -> Dict:
        """Parse query and search PubMed"""
        try:
            data = {"query": query, "max_articles": max_articles}
            response = self.session.post(f"{self.base_url}/query/parse", json=data, timeout=30)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_data(self, query_id: str, sample_size: Optional[int] = None, 
                     age_range: Optional[List[int]] = None, use_research_context: bool = True,
                     filters: Optional[Dict] = None) -> Dict:
        """Generate synthetic data"""
        try:
            data = {
                "query_id": query_id,
                "use_research_context": use_research_context
            }
            if sample_size:
                data["sample_size"] = sample_size
            if age_range:
                data["age_range"] = age_range
            
            # Add filters if provided
            if filters:
                data.update({
                    "gender": filters.get("gender"),
                    "diagnoses": filters.get("diagnoses"),
                    "diagnosis_logic": filters.get("diagnosis_logic"),
                    "icu_required": filters.get("icu_required"),
                    "mortality": filters.get("mortality"),
                    "risk_level": filters.get("risk_level"),
                    "complexity": filters.get("complexity")
                })
                # Use age_range from filters if not already set by age_range parameter
                if not age_range and filters.get("age_range"):
                    data["age_range"] = filters.get("age_range")
            
            response = self.session.post(f"{self.base_url}/data/generate", json=data, timeout=10)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_generation_status(self, task_id: str) -> Dict:
        """Get generation status"""
        try:
            response = self.session.get(f"{self.base_url}/data/status/{task_id}", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def download_data(self, task_id: str) -> Dict:
        """Download generated data"""
        try:
            response = self.session.get(f"{self.base_url}/data/download/{task_id}", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def search_articles(self, query_id: str, search_query: str, top_k: int = 10) -> Dict:
        """Search articles"""
        try:
            params = {"query": search_query, "top_k": top_k}
            response = self.session.get(f"{self.base_url}/articles/search/{query_id}", params=params, timeout=10)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_examples(self) -> Dict:
        """Get example queries"""
        try:
            response = self.session.get(f"{self.base_url}/examples", timeout=5)
            return response.json()
        except Exception as e:
            return {"examples": []}

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient()

api_client = get_api_client()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #cce4f7;
        border: 1px solid #b8daff;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Compact sidebar metrics */
    .sidebar .metric-container {
        font-size: 0.8rem;
    }
    
    [data-testid="metric-container"] > div > div > div > div {
        font-size: 0.9rem !important;
    }
    
    /* Make status indicators more compact */
    .sidebar [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 0.8rem !important;
    }
    
    /* Compact status text */
    .stMetric > div > div > div > div {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🏥 Medical Research Data Platform</h1>
    <p style="color: white; margin: 0; opacity: 0.9;">Bio.Entrez Enhanced Synthetic Data Generation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for system status and controls
with st.sidebar:
    st.header("🔧 System Status")
    
    # Add a refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Health check with better error handling
    with st.spinner("Checking API status..."):
        health_status = api_client.health_check()
    
    if health_status.get("status") == "healthy":
        st.success("✅ API is healthy")
        
        # System details
        try:
            with st.spinner("Loading system details..."):
                system_status = api_client.get_system_status()
            
            if not system_status.get("error"):
                col1, col2 = st.columns(2)
                with col1:
                    rag_ready = system_status.get("rag_system") == "ready"
                    rag_status = "🟢" if rag_ready else "🔴"
                    st.metric("RAG System", rag_status)
                    
                    ctgan_ready = system_status.get("ctgan_model") == "ready"
                    ctgan_status = "🟢" if ctgan_ready else "🔴"
                    st.metric("CTGAN Model", ctgan_status)
                
                with col2:
                    pubmed_ready = system_status.get("pubmed_connection") == "ready"
                    pubmed_status = "🟢" if pubmed_ready else "🔴"
                    st.metric("PubMed", pubmed_status)
                    
                    articles_count = system_status.get("articles_indexed", 0)
                    st.metric("Articles Indexed", articles_count)
                
                st.metric("Queries Processed", system_status.get("total_queries_processed", 0))
                st.metric("Data Generated", system_status.get("total_data_generated", 0))
            else:
                st.warning("⚠️ Could not load system details")
                st.write(f"Error: {system_status.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.warning("⚠️ Could not load system details")
            st.write(f"Error: {str(e)}")
        
    else:
        st.error("❌ API is not available")
        st.write("**Connection Details:**")
        st.write(f"- API URL: `{API_BASE_URL}`")
        st.write(f"- Error: {health_status.get('message', 'Unknown error')}")
        st.write("**Troubleshooting:**")
        st.write("1. Ensure FastAPI is running: `python api.py`")
        st.write("2. Check if port 8000 is accessible")
        st.write("3. Verify no firewall is blocking the connection")
        st.write("4. Try refreshing this page")
        
        # Add connection test button
        if st.button("🔍 Test Connection"):
            with st.spinner("Testing connection..."):
                import urllib.request
                try:
                    urllib.request.urlopen(f"{API_BASE_URL}/health", timeout=5)
                    st.success("✅ Connection test successful!")
                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")
        
        # Don't stop the app, just show limited functionality
        st.warning("⚠️ Limited functionality available without API connection")
    
    st.markdown("---")
    
    # Settings
    st.header("⚙️ Settings")
    max_articles = st.slider("Max PubMed Articles", 10, 100, 50)
    
    # Store use_research_context in session state
    st.session_state.use_research_context = st.toggle("Use Research Context", value=st.session_state.use_research_context)
    
    # Clear cache button
    if st.button("🗑️ Clear Cache"):
        try:
            response = requests.delete(f"{API_BASE_URL}/system/clear-cache")
            if response.status_code == 200:
                st.success("Cache cleared successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error clearing cache: {e}")

# Main content area
# Add connection status check
connection_ok = health_status.get("status") == "healthy"

if not connection_ok:
    st.error("🚨 **API Connection Issue**")
    st.write(f"Cannot connect to the FastAPI backend at `{API_BASE_URL}`")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Retry Connection"):
            st.rerun()
    
    with col2:
        if st.button("🔍 Test Connection"):
            test_url = f"{API_BASE_URL}/health"
            try:
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200:
                    st.success("✅ Connection successful! Please refresh the page.")
                else:
                    st.error(f"❌ Got HTTP {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
    
    with col3:
        if st.button("📖 Show Help"):
            st.info("""
            **Connection Troubleshooting:**
            1. Make sure FastAPI is running: `python api.py`
            2. Check if you can access: http://backend:8000/health
            3. Verify no firewall is blocking port 8000
            4. Restart both services if needed
            """)
    
    st.markdown("---")
    st.warning("⚠️ **Limited Mode**: Some features may not work without API connection")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query Builder", "📊 Data Generation", "📚 Article Search", "📈 Analytics"])

with tab1:
    st.header("🔍 Research Query Builder")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Query Input")
        
        # Example queries
        examples = api_client.get_examples()
        if examples.get("examples"):
            selected_example = st.selectbox(
                "Choose an example query:",
                [""] + [ex["title"] for ex in examples["examples"]],
                format_func=lambda x: "Select an example..." if x == "" else x
            )
            
            if selected_example:
                example_data = next(ex for ex in examples["examples"] if ex["title"] == selected_example)
                st.info(f"**Description:** {example_data['description']}")
                query_text = example_data["query"]
            else:
                query_text = ""
        else:
            query_text = ""
        
        # Query input
        query = st.text_area(
            "Enter your research query:",
            value=query_text,
            height=100,
            placeholder="e.g., Generate 200 cardiovascular patients aged 60-80 for mortality research"
        )
        
        col_parse, col_context = st.columns(2)
        
        with col_parse:
            parse_button = st.button("🔍 Parse Query", type="primary")
        
        with col_context:
            context_button = st.button("📋 Generate Context", type="secondary")
    
    with col2:
        st.subheader("Quick Settings")
        
        # Override settings - store in session state
        st.session_state.sample_size_override = st.number_input(
            "Sample Size Override", 
            min_value=1, 
            max_value=1000, 
            value=st.session_state.sample_size_override
        )
        
        age_override = st.checkbox("Override Age Range")
        if age_override:
            age_min = st.number_input("Min Age", min_value=18, max_value=100, value=18)
            age_max = st.number_input("Max Age", min_value=18, max_value=100, value=100)
            st.session_state.age_range_override = [age_min, age_max]
        else:
            st.session_state.age_range_override = None
    
    # Process query
    if query and (parse_button or context_button):
        if parse_button:
            with st.spinner("🔍 Parsing query and searching PubMed..."):
                result = api_client.parse_query(query, max_articles)
        else:
            with st.spinner("📋 Generating research context..."):
                result = api_client.generate_context(query, max_articles)
        
        if result.get("success"):
            # Store in session state
            if parse_button:
                st.session_state.query_id = result["query_id"]
                st.session_state.query_result = result
                
                # Extract sample size from parsed parameters if available
                extracted_params = result.get("extracted_params", {})
                if "sample_size" in extracted_params and extracted_params["sample_size"]:
                    st.session_state.sample_size_override = extracted_params["sample_size"]
                    st.success(f"✅ Extracted sample size: {extracted_params['sample_size']}")
                
            else:
                st.session_state.context_id = result.get("query_id", "")
                st.session_state.context_result = result
            
            st.success("✅ Query processed successfully!")
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Extracted Parameters")
                
                if parse_button:
                    params = result["extracted_params"]
                    st.json(params)
                else:
                    params = result.get("clinical_parameters", result.get("extracted_params", {}))
                    st.json(params)
            
            with col2:
                st.subheader("📚 PubMed Results")
                
                if parse_button:
                    articles = result.get("pubmed_results", [])
                    st.metric("Articles Found", len(articles))
                    st.metric("Embeddings Created", result.get("embeddings_created", 0))
                else:
                    st.metric("Articles Retrieved", len(result.get("pubmed_results", [])))
                    st.metric("Key Findings", len(result.get("key_findings", [])))
            
            # Show research context
            st.subheader("📋 Research Context")
            
            if parse_button:
                context = result.get("research_context", "No research context available.")
            else:
                context = result.get("research_context", "No research context available.")
            
            with st.expander("View Research Context", expanded=True):
                st.text(context)
            
            # Show articles
            if parse_button and result.get("pubmed_results", []):
                st.subheader("📄 Top Articles")
                
                for i, article in enumerate(result.get("pubmed_results", [])[:5], 1):
                    with st.expander(f"{i}. {article.get('title', 'No title available')}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Journal:** {article.get('journal', 'Unknown')}")
                            st.write(f"**Date:** {article.get('publication_date', article.get('year', 'Unknown'))}")
                            st.write(f"**Abstract:** {article.get('abstract', 'No abstract available')}")
                        
                        with col2:
                            st.metric("PMID", article.get('pmid', 'N/A'))
                            st.metric("Relevance", f"{article.get('relevance_score', 0):.2f}")
            
            # Show key findings for context generation
            if context_button and result.get("key_findings"):
                st.subheader("🔍 Key Findings")
                
                for i, finding in enumerate(result.get("key_findings", []), 1):
                    st.write(f"**{i}.** {finding}")
        
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

with tab2:
    st.header("📊 Synthetic Data Generation")
    
    # Check if we have a query to work with
    if not (hasattr(st.session_state, 'query_id') or hasattr(st.session_state, 'context_id')):
        st.info("👈 Please parse a query or generate context first in the Query Builder tab.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔄 Generate Synthetic Data")
            
            # Show which query/context we're using
            if hasattr(st.session_state, 'query_id'):
                st.success(f"Using Query ID: `{st.session_state.query_id}`")
                current_id = st.session_state.query_id
            else:
                st.success(f"Using Context ID: `{st.session_state.context_id}`")
                current_id = st.session_state.context_id
            
            # Generation parameters - use session state values
            sample_size = st.number_input(
                "Sample Size", 
                min_value=1, 
                max_value=1000, 
                value=st.session_state.sample_size_override,
                help="This value can be set in the Query Builder tab or overridden here"
            )
            
            # Show if age range override is active
            if st.session_state.age_range_override:
                st.info(f"Age range override active: {st.session_state.age_range_override[0]}-{st.session_state.age_range_override[1]} years")
            
            # Show research context setting
            st.info(f"Research context: {'Enabled' if st.session_state.use_research_context else 'Disabled'}")
            
            # Show active filters if available
            if hasattr(st.session_state, 'query_result') and st.session_state.query_result:
                extracted_params = st.session_state.query_result.get("extracted_params", {})
                active_filters = []
                
                if extracted_params.get("gender"):
                    active_filters.append(f"Gender: {extracted_params['gender']}")
                if extracted_params.get("diagnoses"):
                    logic = extracted_params.get("diagnosis_logic", "OR")
                    diagnoses_str = f" {logic} ".join(extracted_params['diagnoses'])
                    active_filters.append(f"Diagnoses: {diagnoses_str}")
                if extracted_params.get("age_range"):
                    min_age, max_age = extracted_params["age_range"]
                    active_filters.append(f"Age: {min_age}-{max_age}")
                if extracted_params.get("icu_required"):
                    active_filters.append("ICU Required: Yes")
                if extracted_params.get("mortality") is not None:
                    active_filters.append(f"Mortality: {'Yes' if extracted_params['mortality'] else 'No'}")
                if extracted_params.get("risk_level"):
                    active_filters.append(f"Risk Level: {extracted_params['risk_level']}")
                if extracted_params.get("complexity"):
                    active_filters.append(f"Complexity: {extracted_params['complexity']}")
                
                if active_filters:
                    st.success(f"🎯 Active Filters: {' | '.join(active_filters)}")
                    
                    # Check for restrictive filter combinations
                    has_young_age = st.session_state.age_range_override and st.session_state.age_range_override[1] <= 35
                    has_high_risk = extracted_params.get("risk_level") == "high"
                    has_high_complexity = extracted_params.get("complexity") == "high"
                    
                    if has_young_age and (has_high_risk or has_high_complexity):
                        st.warning("⚠️ **Restrictive Filter Warning**: Young patients with high risk/complexity are rare in clinical data. You may receive fewer patients than requested. Consider widening age range or reducing filter constraints for larger datasets.")
                    elif has_high_risk and has_high_complexity:
                        st.info("ℹ️ **Filter Notice**: High-risk + high-complexity patients are uncommon. The system will generate as many matching patients as possible from the training data.")
                else:
                    st.info("📊 No specific filters detected - generating general population data")
            
            if st.button("🚀 Generate Data", type="primary"):
                with st.spinner("🔄 Starting data generation..."):
                    # Extract filters from the query result if available
                    filters = None
                    if hasattr(st.session_state, 'query_result') and st.session_state.query_result:
                        extracted_params = st.session_state.query_result.get("extracted_params", {})
                        filters = {
                            "gender": extracted_params.get("gender"),
                            "diagnoses": extracted_params.get("diagnoses"),
                            "diagnosis_logic": extracted_params.get("diagnosis_logic"),
                            "icu_required": extracted_params.get("icu_required"),
                            "mortality": extracted_params.get("mortality"),
                            "risk_level": extracted_params.get("risk_level"),
                            "complexity": extracted_params.get("complexity"),
                            "age_range": extracted_params.get("age_range")
                        }
                    
                    gen_result = api_client.generate_data(
                        current_id,
                        sample_size=sample_size,
                        age_range=st.session_state.age_range_override,
                        use_research_context=st.session_state.use_research_context,
                        filters=filters
                    )
                
                if gen_result.get("success"):
                    task_id = gen_result.get("task_id", "")
                    st.session_state.task_id = task_id
                    
                    st.success(f"✅ Generation started! Task ID: `{task_id}`")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Poll for completion
                    max_attempts = 60  # 5 minutes max
                    attempt = 0
                    
                    while attempt < max_attempts:
                        status = api_client.get_generation_status(task_id)
                        
                        if status.get("status") == "completed":
                            progress_bar.progress(100)
                            status_text.success("✅ Generation completed!")
                            
                            # Download results
                            data_result = api_client.download_data(task_id)
                            st.session_state.generated_data = data_result
                            
                            break
                        elif status.get("status") == "failed":
                            progress_bar.progress(0)
                            status_text.error(f"❌ Generation failed: {status.get('error', 'Unknown error')}")
                            break
                        else:
                            progress = min(90, (attempt / max_attempts) * 100)
                            progress_bar.progress(int(progress))
                            status_text.info(f"⏳ Status: {status.get('status', 'processing')} - {status.get('progress', 'Working...')}")
                            
                            time.sleep(5)
                            attempt += 1
                    
                    if attempt >= max_attempts:
                        status_text.error("⏰ Generation timed out. Please check the task status manually.")
                
                else:
                    st.error(f"❌ Error starting generation: {gen_result.get('error', 'Unknown error')}")
        
        with col2:
            st.subheader("📈 Generation Status")
            
            # Show current settings
            st.info(f"**Sample Size:** {st.session_state.sample_size_override}")
            if st.session_state.age_range_override:
                st.info(f"**Age Range:** {st.session_state.age_range_override[0]}-{st.session_state.age_range_override[1]}")
            st.info(f"**Research Context:** {'On' if st.session_state.use_research_context else 'Off'}")
            
            if hasattr(st.session_state, 'task_id'):
                if st.button("🔄 Refresh Status"):
                    status = api_client.get_generation_status(st.session_state.task_id)
                    st.json(status)
            else:
                st.info("No active generation task")
    
    # Show results if available
    if hasattr(st.session_state, 'generated_data') and st.session_state.generated_data:
        st.subheader("📊 Generated Data Results")
        
        data_result = st.session_state.generated_data
        metadata = data_result.get("metadata", {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", metadata.get("generated_count", metadata.get("total_patients", 0)))
        
        with col2:
            avg_age = metadata.get('avg_age', 0)
            age_std = metadata.get('age_std', 0)
            st.metric("Average Age", f"{avg_age:.1f} ± {age_std:.1f}")
        
        with col3:
            icu_rate = metadata.get('icu_rate', 0)
            st.metric("ICU Rate", f"{icu_rate:.1%}")
        
        with col4:
            mortality_rate = metadata.get('mortality_rate', 0)
            st.metric("Mortality Rate", f"{mortality_rate:.1%}")
        
        # Data preview and downloads
        if "data" in data_result and data_result.get("data"):
            df = pd.DataFrame(data_result["data"])
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download options
            st.subheader("💾 Download Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = json.dumps(data_result, indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col3:
                metadata_json = json.dumps(metadata, indent=2)
                st.download_button(
                    label="📈 Download Metadata",
                    data=metadata_json,
                    file_name=f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

with tab3:
    st.header("📚 Article Search")
    
    if not hasattr(st.session_state, 'query_id'):
        st.info("👈 Please parse a query first in the Query Builder tab to enable article search.")
    else:
        search_query = st.text_input(
            "Search through retrieved articles:",
            placeholder="e.g., treatment effectiveness, mortality risk factors, clinical outcomes"
        )
        
        top_k = st.slider("Number of results", 1, 20, 10)
        
        if st.button("🔍 Search Articles") and search_query:
            with st.spinner("🔍 Searching articles..."):
                search_result = api_client.search_articles(st.session_state.query_id, search_query, top_k)
            
            if search_result.get("success"):
                st.success(f"✅ Found {search_result['articles_found']} relevant articles")
                
                articles = search_result.get("articles", [])
                
                for i, article in enumerate(articles, 1):
                    with st.expander(f"{i}. {article['title']} (Score: {article['similarity_score']:.3f})"):
                        st.write(f"**Journal:** {article['journal']}")
                        st.write(f"**Date:** {article['publication_date']}")
                        st.write(f"**Abstract:** {article.get('abstract', 'No abstract available')}")
            else:
                st.error(f"❌ Search failed: {search_result.get('error', 'Unknown error')}")

with tab4:
    st.header("📈 System Analytics")
    
    # System metrics
    system_status = api_client.get_system_status()
    
    if not system_status.get("error"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Queries", system_status.get("total_queries_processed", 0))
            st.metric("📄 Articles Indexed", system_status.get("articles_indexed", 0))
        
        with col2:
            st.metric("🔬 Data Generated", system_status.get("total_data_generated", 0))
            st.metric("⚙️ Active Tasks", system_status.get("active_generation_tasks", 0))
        
        with col3:
            st.metric("🤖 RAG System", "🟢 Ready" if system_status.get("rag_system") == "ready" else "🔴 Not Ready")
            st.metric("🏥 PubMed", "🟢 Ready" if system_status.get("pubmed_connection") == "ready" else "🔴 Error")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>🏥 Dr. Freak </p>
</div>
""", unsafe_allow_html=True)