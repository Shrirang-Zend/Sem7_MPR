#!/usr/bin/env python3
"""
Documentation Page - API documentation and usage examples
"""

import streamlit as st
import requests
import json
import sys
from pathlib import Path

# Add frontend path for importing components
frontend_path = Path(__file__).parent.parent
sys.path.insert(0, str(frontend_path))

# Add backend path for importing config - handle both local and container environments
backend_path = Path(__file__).parent.parent.parent / "backend"
if not backend_path.exists():
    # Try container path structure
    backend_path = Path("/app/backend")
sys.path.insert(0, str(backend_path))

from components.api_client import APIClient

# Import API_CONFIG with fallback
try:
    from config.settings import API_CONFIG
except ImportError:
    # Fallback configuration for when config is not available
    API_CONFIG = {
        'host': 'localhost',
        'port': 8000,
        'max_generated_rows': 2000,
        'timeout_seconds': 300
    }

st.set_page_config(
    page_title="Documentation",
    page_icon="📚",
    layout="wide"
)

st.title("📚 API Documentation & Usage Guide")
st.markdown("Complete guide to using the Healthcare Data Generation API")

# Initialize API client for testing
# Use the same URL as the main app for consistency
API_BASE_URL = "http://backend:8000"  # Docker service name

# Sidebar navigation
with st.sidebar:
    st.header("📖 Navigation")
    
    sections = [
        "🚀 Quick Start",
        "🔌 API Endpoints",
        "📝 Request Examples",
        "📊 Response Formats",
        "❌ Error Handling",
        "💡 Best Practices"
    ]
    
    selected_section = st.radio("Select Section:", sections)

# Main content based on selection
if selected_section == "🚀 Quick Start":
    st.header("🚀 Quick Start Guide")
    
    st.markdown("""
    ## Getting Started
    
    The Healthcare Data Generation API allows you to generate synthetic healthcare data using a trained CTGAN model.
    
    ### Base URL
    ```
    http://backend:8000
    ```
    
    ### Authentication
    Currently, no authentication is required for local development.
    
    ### Content Type
    All requests should use `Content-Type: application/json`
    """)
    
    st.subheader("🔥 Quick Example")
    
    st.code("""
import requests

# Generate 10 patients with diabetes
response = requests.post('http://backend:8000/generate', json={
    "query": "Generate patients with diabetes",
    "num_patients": 10,
    "format": "json"
})

data = response.json()
print(f"Generated {len(data['data'])} patients")
    """, language="python")
    
    st.subheader("✅ Test Connection")
    
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API is accessible!")
                st.json(response.json())
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
            st.info("Make sure the API server is running: `python scripts/08_run_api.py`")

elif selected_section == "🔌 API Endpoints":
    st.header("🔌 API Endpoints")
    
    endpoints = [
        {
            "method": "GET",
            "path": "/health",
            "description": "Check API health status",
            "parameters": "None",
            "example": "GET /health"
        },
        {
            "method": "GET", 
            "path": "/statistics",
            "description": "Get dataset statistics and model information",
            "parameters": "None",
            "example": "GET /statistics"
        },
        {
            "method": "POST",
            "path": "/generate",
            "description": "Generate synthetic healthcare data",
            "parameters": "query, num_patients, format",
            "example": "POST /generate"
        }
    ]
    
    for endpoint in endpoints:
        st.subheader(f"{endpoint['method']} {endpoint['path']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Description:** {endpoint['description']}")
            st.write(f"**Parameters:** {endpoint['parameters']}")
            
            if st.button(f"Test {endpoint['path']}", key=f"test_{endpoint['path']}"):
                try:
                    if endpoint['method'] == 'GET':
                        response = requests.get(f"{API_BASE_URL}{endpoint['path']}", timeout=10)
                    else:
                        # Use sample data for POST
                        sample_data = {
                            "query": "Generate sample patients",
                            "num_patients": 5,
                            "format": "json"
                        }
                        response = requests.post(f"{API_BASE_URL}{endpoint['path']}", 
                                               json=sample_data, timeout=30)
                    
                    st.write(f"**Status:** {response.status_code}")
                    if response.status_code == 200:
                        st.json(response.json())
                    else:
                        st.error(response.text)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            st.code(f"""
# Example {endpoint['method']} request
curl -X {endpoint['method']} \\
  {API_BASE_URL}{endpoint['path']} \\
  -H 'Content-Type: application/json'""" + ("""\\
  -d '{
    "query": "Generate patients with diabetes",
    "num_patients": 10,
    "format": "json"
  }'""" if endpoint['method'] == 'POST' else ""), language="bash")
        
        st.markdown("---")

elif selected_section == "📝 Request Examples":
    st.header("📝 Request Examples")
    
    st.subheader("1. Basic Data Generation")
    
    st.code("""
POST /generate
Content-Type: application/json

{
  "query": "Generate 20 patients with various conditions",
  "num_patients": 20,
  "format": "json"
}
    """, language="json")
    
    st.subheader("2. Specific Medical Conditions")
    
    st.code("""
POST /generate
Content-Type: application/json

{
  "query": "Generate elderly patients with cardiovascular disease",
  "num_patients": 15,
  "format": "csv"
}
    """, language="json")
    
    st.subheader("3. ICU Patients")
    
    st.code("""
POST /generate
Content-Type: application/json

{
  "query": "Generate ICU patients with respiratory complications",
  "num_patients": 25,
  "format": "json"
}
    """, language="json")
    
    st.subheader("4. Pediatric Cases")
    
    st.code("""
POST /generate
Content-Type: application/json

{
  "query": "Generate pediatric patients under 18 years old",
  "num_patients": 30,
  "format": "json"
}
    """, language="json")
    
    # Interactive example
    st.subheader("🧪 Try Your Own Request")
    
    with st.form("custom_request"):
        query = st.text_area(
            "Query Description:",
            "Generate patients with diabetes and hypertension",
            help="Describe the type of patients you want to generate"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            num_patients = st.number_input("Number of Patients:", 1, 100, 10)
        with col2:
            format_type = st.selectbox("Format:", ["json", "csv"])
        
        submitted = st.form_submit_button("🚀 Send Request")
        
        if submitted:
            try:
                payload = {
                    "query": query,
                    "num_patients": num_patients,
                    "format": format_type
                }
                
                with st.spinner("Generating data..."):
                    response = requests.post(
                        f"{API_BASE_URL}/generate",
                        json=payload,
                        timeout=30
                    )
                
                st.write("**Request:**")
                st.json(payload)
                
                st.write("**Response:**")
                if response.status_code == 200:
                    st.success(f"✅ Success! Status: {response.status_code}")
                    result = response.json()
                    st.json(result)
                else:
                    st.error(f"❌ Error! Status: {response.status_code}")
                    st.text(response.text)
                    
            except Exception as e:
                st.error(f"Request failed: {str(e)}")

elif selected_section == "📊 Response Formats":
    st.header("📊 Response Formats")
    
    st.subheader("✅ Successful Response")
    
    st.code("""
{
  "success": true,
  "message": "Data generated successfully",
  "data": [
    {
      "patient_id": "P001",
      "age": 67,
      "gender": "M",
      "diagnoses": "Diabetes mellitus, Hypertension",
      "hospital_los_days": 5.2,
      "icu_los_days": 0.0,
      "has_icu_stay": false,
      "mortality": false,
      "emergency_admission": true,
      "high_complexity": false
    }
  ],
  "metadata": {
    "total_patients": 1,
    "avg_age": 67.0,
    "gender_ratio": {"M": 1.0, "F": 0.0},
    "icu_rate": 0.0,
    "mortality_rate": 0.0,
    "quality_score": 0.95,
    "generation_time_seconds": 2.1
  }
}
    """, language="json")
    
    st.subheader("❌ Error Response")
    
    st.code("""
{
  "success": false,
  "error": "Invalid number of patients. Must be between 1 and 1000.",
  "details": {
    "error_code": "INVALID_PARAMETER",
    "parameter": "num_patients",
    "provided_value": 1500,
    "valid_range": "1-1000"
  }
}
    """, language="json")
    
    st.subheader("📈 Statistics Response")
    
    st.code("""
{
  "total_patients": 100000,
  "avg_age": 45.6,
  "unique_diagnoses": 250,
  "icu_rate": 0.15,
  "mortality_rate": 0.08,
  "model_accuracy": 0.94,
  "quality_score": 0.91,
  "diagnosis_distribution": {
    "Diabetes mellitus": 1500,
    "Hypertension": 1200,
    "Pneumonia": 800
  },
  "age_distribution": {
    "0-10": 500,
    "11-20": 600,
    "21-30": 800
  }
}
    """, language="json")

elif selected_section == "❌ Error Handling":
    st.header("❌ Error Handling")
    
    st.subheader("📋 Common Error Codes")
    
    errors = [
        {
            "code": 400,
            "name": "Bad Request",
            "description": "Invalid request parameters",
            "example": "Missing required field 'query'"
        },
        {
            "code": 422,
            "name": "Validation Error", 
            "description": "Parameter validation failed",
            "example": "num_patients must be between 1 and 1000"
        },
        {
            "code": 500,
            "name": "Internal Server Error",
            "description": "Model generation failed",
            "example": "CTGAN model encountered an error"
        },
        {
            "code": 503,
            "name": "Service Unavailable",
            "description": "API temporarily unavailable",
            "example": "Model is currently being retrained"
        }
    ]
    
    for error in errors:
        st.write(f"**{error['code']} - {error['name']}**")
        st.write(f"Description: {error['description']}")
        st.write(f"Example: {error['example']}")
        st.markdown("---")
    
    st.subheader("🛠️ Error Handling Best Practices")
    
    st.code("""
import requests
import time

def generate_healthcare_data(query, num_patients, max_retries=3):
    url = "http://backend:8000/generate"
    payload = {
        "query": query,
        "num_patients": num_patients,
        "format": "json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 422:
                # Validation error - don't retry
                print(f"Validation error: {response.text}")
                return None
            elif response.status_code >= 500:
                # Server error - retry with backoff
                wait_time = 2 ** attempt
                print(f"Server error, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Unexpected error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
        except requests.exceptions.ConnectionError:
            print("Connection error - is the API running?")
            return None
    
    print("Max retries exceeded")
    return None
    """, language="python")

elif selected_section == "💡 Best Practices":
    st.header("💡 Best Practices")
    
    st.subheader("🎯 Query Optimization")
    
    st.markdown("""
    ### Effective Query Writing
    
    **Good Examples:**
    - "Generate elderly patients with diabetes and cardiovascular disease"
    - "Create ICU patients with respiratory failure requiring ventilation"
    - "Generate pediatric patients with congenital conditions"
    
    **Avoid:**
    - Vague queries: "Generate some patients"
    - Contradictory requirements: "Generate healthy patients with severe illness"
    - Too many specific constraints that may not exist in training data
    """)
    
    st.subheader("📊 Data Size Considerations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Small Datasets (1-50 patients):**
        - Fast generation (< 5 seconds)
        - Good for testing and prototyping
        - Immediate analysis possible
        """)
    
    with col2:
        st.markdown("""
        **Large Datasets (100-1000 patients):**
        - Longer generation time (10-60 seconds)
        - Better statistical representation
        - Consider pagination for analysis
        """)
    
    st.subheader("🔄 Rate Limiting")
    
    st.markdown("""
    ### Recommended Usage Patterns
    
    - **Development:** Up to 10 requests per minute
    - **Testing:** Batch requests with delays
    - **Production:** Implement exponential backoff
    
    ### Code Example:
    """)
    
    st.code("""
import time
from typing import List, Dict

def batch_generate(queries: List[str], delay: float = 1.0) -> List[Dict]:
    results = []
    
    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(delay)  # Rate limiting
        
        result = generate_healthcare_data(query, 10)
        if result:
            results.append(result)
        
        # Progress feedback
        print(f"Completed {i+1}/{len(queries)} requests")
    
    return results
    """, language="python")
    
    st.subheader("💾 Data Validation")
    
    st.markdown("""
    ### Always Validate Generated Data
    
    1. **Check Response Status:** Verify `success: true`
    2. **Validate Data Structure:** Ensure expected fields are present
    3. **Quality Metrics:** Review `quality_score` in metadata
    4. **Clinical Validity:** Check for medically reasonable values
    """)
    
    st.code("""
def validate_generated_data(response_data):
    if not response_data.get('success'):
        raise ValueError(f"Generation failed: {response_data.get('error')}")
    
    data = response_data.get('data', [])
    if not data:
        raise ValueError("No data generated")
    
    metadata = response_data.get('metadata', {})
    quality_score = metadata.get('quality_score', 0)
    
    if quality_score < 0.8:
        print(f"Warning: Low quality score ({quality_score:.2f})")
    
    # Validate expected fields
    required_fields = ['patient_id', 'age', 'gender']
    for patient in data:
        for field in required_fields:
            if field not in patient:
                raise ValueError(f"Missing required field: {field}")
    
    return True
    """, language="python")

# Footer
st.markdown("---")
st.markdown("*Healthcare Data Generation API Documentation v1.0*")
st.markdown("For technical support, check the system logs or contact your administrator.")