"""
query_parser.py

Natural language query parser for healthcare data generation.

This module provides functions to parse natural language queries into
structured filters for synthetic data generation.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class HealthcareQueryParser:
    """
    Parses natural language queries for healthcare data generation.
    """
    
    def __init__(self):
        self.diagnosis_keywords = {
            'DIABETES': ['diabetes', 'diabetic', 'dm', 'glucose', 'insulin', 'hyperglycemia'],
            'CARDIOVASCULAR': ['cardiovascular', 'cardiac', 'heart', 'cvd', 'coronary', 'myocardial', 'angina'],
            'HYPERTENSION': ['hypertension', 'hypertensive', 'high blood pressure', 'htn', 'bp'],
            'RENAL': ['renal', 'kidney', 'nephritis', 'dialysis', 'ckd', 'chronic kidney'],
            'RESPIRATORY': ['respiratory', 'copd', 'asthma', 'pneumonia', 'lung', 'bronchitis', 'pulmonary'],
            'SEPSIS': ['sepsis', 'septic', 'infection', 'bacteremia', 'septicemia'],
            'NEUROLOGICAL': ['neurological', 'stroke', 'seizure', 'brain', 'neuro', 'cerebral'],
            'TRAUMA': ['trauma', 'fracture', 'injury', 'accident', 'wound', 'broken'],
            'CANCER': ['cancer', 'tumor', 'malignant', 'oncology', 'carcinoma', 'neoplasm'],
            'OTHER': ['other', 'miscellaneous', 'unspecified']
        }
        
        self.age_patterns = {
            'elderly': (65, 100),
            'geriatric': (80, 100),
            'senior': (65, 100),
            'old': (65, 100),
            'middle-aged': (40, 65),
            'adult': (18, 65),
            'young': (18, 40),
            'pediatric': (0, 18),
            'child': (0, 18),
            'infant': (0, 2)
        }
        
        self.gender_patterns = {
            'female': ['female', 'women', 'woman', 'girl'],
            'male': ['male', 'men', 'man', 'boy']
        }
        
        self.severity_patterns = {
            'critical': ['critical', 'severe', 'life-threatening', 'acute'],
            'high': ['high', 'serious', 'major'],
            'medium': ['moderate', 'medium', 'intermediate'],
            'low': ['mild', 'minor', 'low']
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured filters.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary of parsed filters
        """
        query_lower = query.lower().strip()
        
        filters = {
            'diagnoses': [],
            'age_range': None,
            'gender': None,
            'icu_required': None,
            'mortality': None,
            'risk_level': None,
            'complexity': None,
            'emergency': None,
            'length_of_stay': None
        }
        
        # Extract diagnoses
        filters['diagnoses'] = self._extract_diagnoses(query_lower)
        
        # Extract age information
        filters['age_range'] = self._extract_age_range(query_lower)
        
        # Extract gender
        filters['gender'] = self._extract_gender(query_lower)
        
        # Extract ICU requirement
        filters['icu_required'] = self._extract_icu_requirement(query_lower)
        
        # Extract mortality information
        filters['mortality'] = self._extract_mortality(query_lower)
        
        # Extract risk level
        filters['risk_level'] = self._extract_risk_level(query_lower)
        
        # Extract complexity
        filters['complexity'] = self._extract_complexity(query_lower)
        
        # Extract emergency status
        filters['emergency'] = self._extract_emergency_status(query_lower)
        
        # Extract length of stay preferences
        filters['length_of_stay'] = self._extract_length_of_stay(query_lower)
        
        logger.debug(f"Parsed query '{query}' into filters: {filters}")
        return filters
    
    def _extract_diagnoses(self, query: str) -> List[str]:
        """Extract diagnosis conditions from query."""
        found_diagnoses = []
        
        for diagnosis, keywords in self.diagnosis_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_diagnoses.append(diagnosis)
        
        return found_diagnoses
    
    def _extract_age_range(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract age range from query."""
        # Check for specific age patterns
        for age_term, age_range in self.age_patterns.items():
            if age_term in query:
                return age_range
        
        # Check for specific age numbers
        age_match = re.search(r'age[s]?\s*(\d+)', query)
        if age_match:
            age = int(age_match.group(1))
            return (age - 5, age + 5)  # Range around specific age
        
        # Check for age ranges like "18-65" or "over 70"
        range_match = re.search(r'(\d+)\s*[-to]\s*(\d+)', query)
        if range_match:
            min_age = int(range_match.group(1))
            max_age = int(range_match.group(2))
            return (min_age, max_age)
        
        over_match = re.search(r'over\s*(\d+)', query)
        if over_match:
            min_age = int(over_match.group(1))
            return (min_age, 100)
        
        under_match = re.search(r'under\s*(\d+)', query)
        if under_match:
            max_age = int(under_match.group(1))
            return (18, max_age)
        
        return None
    
    def _extract_gender(self, query: str) -> Optional[str]:
        """Extract gender preference from query."""
        for gender, keywords in self.gender_patterns.items():
            if any(keyword in query for keyword in keywords):
                return gender
        return None
    
    def _extract_icu_requirement(self, query: str) -> Optional[bool]:
        """Extract ICU requirement from query."""
        icu_keywords = ['icu', 'intensive care', 'critical care', 'critical unit']
        non_icu_keywords = ['non-icu', 'general ward', 'regular ward']
        
        if any(keyword in query for keyword in icu_keywords):
            return True
        elif any(keyword in query for keyword in non_icu_keywords):
            return False
        
        return None
    
    def _extract_mortality(self, query: str) -> Optional[bool]:
        """Extract mortality information from query."""
        mortality_keywords = ['died', 'death', 'mortality', 'fatal', 'deceased']
        survival_keywords = ['survived', 'survivor', 'discharged', 'recovered']
        
        if any(keyword in query for keyword in mortality_keywords):
            return True
        elif any(keyword in query for keyword in survival_keywords):
            return False
        
        return None
    
    def _extract_risk_level(self, query: str) -> Optional[str]:
        """Extract risk level from query."""
        for risk_level, keywords in self.severity_patterns.items():
            if any(keyword in query for keyword in keywords):
                return risk_level
        return None
    
    def _extract_complexity(self, query: str) -> Optional[str]:
        """Extract complexity level from query."""
        high_complexity_keywords = [
            'complex', 'complicated', 'multiple', 'comorbid', 'comorbidity',
            'multi-diagnosis', 'several conditions'
        ]
        low_complexity_keywords = [
            'simple', 'single', 'straightforward', 'uncomplicated'
        ]
        
        if any(keyword in query for keyword in high_complexity_keywords):
            return 'high'
        elif any(keyword in query for keyword in low_complexity_keywords):
            return 'low'
        
        return None
    
    def _extract_emergency_status(self, query: str) -> Optional[bool]:
        """Extract emergency admission status from query."""
        emergency_keywords = ['emergency', 'urgent', 'acute', 'emergent']
        elective_keywords = ['elective', 'planned', 'scheduled']
        
        if any(keyword in query for keyword in emergency_keywords):
            return True
        elif any(keyword in query for keyword in elective_keywords):
            return False
        
        return None
    
    def _extract_length_of_stay(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract length of stay preferences from query."""
        los_info = {}
        
        # Look for specific day mentions
        days_match = re.search(r'(\d+)\s*days?', query)
        if days_match:
            days = int(days_match.group(1))
            los_info['target_days'] = days
        
        # Look for stay duration indicators
        if 'short stay' in query or 'brief' in query:
            los_info['duration'] = 'short'
        elif 'long stay' in query or 'extended' in query:
            los_info['duration'] = 'long'
        
        return los_info if los_info else None
    
    def extract_count(self, query: str) -> int:
        """
        Extract the number of records requested from query.
        
        Args:
            query: Natural language query string
            
        Returns:
            Number of records to generate (default 100)
        """
        # Look for specific numbers
        count_patterns = [
            r'(\d+)\s*patients?',
            r'(\d+)\s*records?',
            r'(\d+)\s*cases?',
            r'generate\s*(\d+)',
            r'create\s*(\d+)',
            r'(\d+)\s*samples?'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, query.lower())
            if match:
                count = int(match.group(1))
                # Reasonable limits
                return min(max(count, 1), 2000)
        
        # Default count
        return 100
    
    def validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the parsed filters.
        
        Args:
            filters: Dictionary of parsed filters
            
        Returns:
            Validated and cleaned filters
        """
        validated = {}
        
        # Validate diagnoses
        if filters.get('diagnoses'):
            validated['diagnoses'] = [d for d in filters['diagnoses'] if d in self.diagnosis_keywords]
        
        # Validate age range
        if filters.get('age_range'):
            min_age, max_age = filters['age_range']
            min_age = max(0, min_age)
            max_age = min(100, max_age)
            if min_age < max_age:
                validated['age_range'] = (min_age, max_age)
        
        # Validate gender
        if filters.get('gender') in ['male', 'female']:
            validated['gender'] = filters['gender']
        
        # Validate boolean fields
        boolean_fields = ['icu_required', 'mortality', 'emergency']
        for field in boolean_fields:
            if filters.get(field) is not None:
                validated[field] = bool(filters[field])
        
        # Validate categorical fields
        if filters.get('risk_level') in ['low', 'medium', 'high', 'critical']:
            validated['risk_level'] = filters['risk_level']
        
        if filters.get('complexity') in ['low', 'medium', 'high']:
            validated['complexity'] = filters['complexity']
        
        # Validate length of stay
        if filters.get('length_of_stay'):
            los = filters['length_of_stay']
            validated_los = {}
            if los.get('target_days') and isinstance(los['target_days'], int):
                validated_los['target_days'] = max(1, min(365, los['target_days']))
            if los.get('duration') in ['short', 'medium', 'long']:
                validated_los['duration'] = los['duration']
            if validated_los:
                validated['length_of_stay'] = validated_los
        
        return validated
    
    def generate_example_queries(self) -> List[str]:
        """
        Generate example queries for API documentation.
        
        Returns:
            List of example query strings
        """
        examples = [
            "Generate 100 elderly patients with diabetes and hypertension",
            "Create 50 ICU patients with cardiovascular disease",
            "Generate young adults with respiratory conditions who survived",
            "Create critical care patients with multiple comorbidities",
            "Generate 200 emergency admission patients",
            "Create elderly female patients with renal disease",
            "Generate patients with short hospital stays",
            "Create high-risk cardiac patients over 70",
            "Generate trauma patients who died",
            "Create complex cases with sepsis and organ failure"
        ]
        return examples