"""
API Client for Healthcare Data Generation System
Handles all API communication with proper error handling and retries
"""

import requests
import time
from typing import Dict, Optional, Any


class APIClient:
    """Client for interacting with the Healthcare Data Generation API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Healthcare-Frontend/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health with retries
        
        Returns:
            Dict containing health status and details
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/health", 
                    timeout=10
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "status": "error", 
                        "message": f"HTTP {response.status_code}: {response.text}"
                    }
                    
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {
                    "status": "error", 
                    "message": "Connection refused - API server not responding"
                }
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {
                    "status": "error", 
                    "message": "Request timeout - API server too slow"
                }
                
            except Exception as e:
                return {
                    "status": "error", 
                    "message": f"Unexpected error: {str(e)}"
                }
        
        return {
            "status": "error", 
            "message": "Max retries exceeded"
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics from the API
        
        Returns:
            Dict containing statistics or error information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/statistics", 
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error - API not available"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def generate_data(self, query: str, num_patients: int = 10, 
                     format_type: str = "json") -> Dict[str, Any]:
        """
        Generate synthetic healthcare data
        
        Args:
            query: Description of the data to generate
            num_patients: Number of patients to generate
            format_type: Output format ('json' or 'csv')
        
        Returns:
            Dict containing generated data or error information
        """
        try:
            payload = {
                "query": query,
                "num_patients": num_patients,
                "format": format_type
            }
            
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=30  # Longer timeout for generation
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Generation timeout - request took too long"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error - API not available"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def test_endpoint(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Test a specific API endpoint
        
        Args:
            endpoint: API endpoint path
            method: HTTP method ('GET' or 'POST')
            data: Request data for POST requests
        
        Returns:
            Dict containing response details
        """
        try:
            url = f"{self.base_url}{endpoint}"
            start_time = time.time()
            
            if method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=30)
            else:
                response = self.session.get(url, timeout=10)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "headers": dict(response.headers),
                "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "status_code": None
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error - API not available",
                "status_code": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "status_code": None
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status from the API
        
        Returns:
            Dict containing system status or error information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/system/status", 
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error - API not available"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def get_examples(self) -> Dict[str, Any]:
        """
        Get example queries from the API
        
        Returns:
            Dict containing examples or error information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/examples", 
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "examples": []
                }
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout", "examples": []}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error - API not available", "examples": []}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}", "examples": []}

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information and available endpoints
        
        Returns:
            Dict containing API information
        """
        endpoints = [
            {
                "path": "/health",
                "method": "GET",
                "description": "Check API health status"
            },
            {
                "path": "/statistics",
                "method": "GET", 
                "description": "Get dataset statistics"
            },
            {
                "path": "/generate",
                "method": "POST",
                "description": "Generate synthetic healthcare data"
            },
            {
                "path": "/system/status",
                "method": "GET",
                "description": "Get detailed system status"
            }
        ]
        
        return {
            "base_url": self.base_url,
            "endpoints": endpoints,
            "version": "1.0",
            "description": "Healthcare Data Generation API"
        }