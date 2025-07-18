"""
PubMed API client for retrieving medical literature
"""

import os
import requests
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PubMedArticle:
    """Represents a PubMed article"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str] = None
    relevance_score: float = 0.0

class PubMedClient:
    """Client for interacting with PubMed E-utilities API"""
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None, tool_name: str = "healthcare_data_generator"):
        # Handle empty string API keys
        api_key_env = os.getenv("PUBMED_API_KEY")
        self.api_key = api_key or (api_key_env if api_key_env and api_key_env.strip() else None)
        self.email = email or os.getenv("PUBMED_EMAIL")
        self.tool_name = tool_name
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': f'{self.tool_name}/1.0'
        })
        
        # Rate limiting - PubMed allows 3 requests per second without API key, 10 with API key
        self.request_delay = 0.1 if self.api_key else 0.34
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """Make a rate-limited request to PubMed API"""
        self._rate_limit()
        
        # Add common parameters
        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email
        params['tool'] = self.tool_name
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {e}")
            raise
    
    def search(self, query: str, max_results: int = 20, sort: str = "relevance") -> List[str]:
        """
        Search PubMed for articles matching the query
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            sort: Sort order (relevance, pub_date, etc.)
            
        Returns:
            List of PMIDs
        """
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'sort': sort,
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request('esearch.fcgi', params)
            
            # Parse XML response
            root = ET.fromstring(response.content)
            pmids = []
            
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
            
            logger.info(f"Found {len(pmids)} PMIDs for query: {query}")
            return pmids
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def fetch_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch detailed information for a list of PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PubMedArticle objects
        """
        if not pmids:
            return []
        
        # PubMed API can handle multiple PMIDs in a single request
        pmid_string = ','.join(pmids)
        
        params = {
            'db': 'pubmed',
            'id': pmid_string,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        try:
            response = self._make_request('efetch.fcgi', params)
            
            # Parse XML response
            root = ET.fromstring(response.content)
            articles = []
            
            for article_elem in root.findall('.//PubmedArticle'):
                try:
                    article = self._parse_article_xml(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing article XML: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(articles)} articles from {len(pmids)} PMIDs")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching article details: {e}")
            return []
    
    def _parse_article_xml(self, article_elem: ET.Element) -> Optional[PubMedArticle]:
        """Parse XML for a single article"""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title available"
            
            # Extract abstract
            abstract_elems = article_elem.findall('.//AbstractText')
            abstract_parts = []
            for elem in abstract_elems:
                if elem.text:
                    label = elem.get('Label', '')
                    text = elem.text
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            
            abstract = ' '.join(abstract_parts) if abstract_parts else "No abstract available"
            
            # Extract authors
            author_elems = article_elem.findall('.//Author')
            authors = []
            for author_elem in author_elems:
                last_name = author_elem.find('.//LastName')
                first_name = author_elem.find('.//ForeName')
                
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{last_name.text} {first_name.text}"
                    authors.append(name)
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown journal"
            
            # Extract year
            year_elem = article_elem.find('.//PubDate/Year')
            year = int(year_elem.text) if year_elem is not None else 0
            
            # Extract DOI
            doi_elem = article_elem.find('.//ArticleId[@IdType="doi"]')
            doi = doi_elem.text if doi_elem is not None else None
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                doi=doi
            )
            
        except Exception as e:
            logger.error(f"Error parsing article XML: {e}")
            return None
    
    def search_and_fetch(self, query: str, max_results: int = 20) -> List[PubMedArticle]:
        """
        Search PubMed and fetch article details in one call
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of PubMedArticle objects
        """
        pmids = self.search(query, max_results)
        if not pmids:
            return []
        
        articles = self.fetch_details(pmids)
        
        # Add relevance scores based on order
        for i, article in enumerate(articles):
            article.relevance_score = 1.0 - (i / len(articles))
        
        return articles
    
    def build_healthcare_query(self, conditions: List[str] = None, demographics: Dict[str, Any] = None) -> str:
        """
        Build a PubMed query optimized for healthcare research
        
        Args:
            conditions: List of medical conditions
            demographics: Demographics info (age, gender, etc.)
            
        Returns:
            Formatted PubMed query string
        """
        query_parts = []
        
        if conditions:
            # Map conditions to proper MeSH terms
            condition_mesh_map = {
                'RESPIRATORY': 'respiratory tract diseases',
                'DIABETES': 'diabetes mellitus',
                'CARDIOVASCULAR': 'cardiovascular diseases',
                'HYPERTENSION': 'hypertension',
                'RENAL': 'kidney diseases',
                'SEPSIS': 'sepsis',
                'NEUROLOGICAL': 'nervous system diseases',
                'TRAUMA': 'wounds and injuries',
                'CANCER': 'neoplasms'
            }
            
            # Add condition terms
            condition_terms = []
            for condition in conditions:
                mesh_term = condition_mesh_map.get(condition, condition.lower())
                # Use both MeSH terms and text words for better coverage
                condition_terms.append(f'("{mesh_term}"[MeSH Terms] OR {mesh_term}[tiab])')
            query_parts.append(f"({' OR '.join(condition_terms)})")
        
        if demographics:
            # Add gender terms
            gender = demographics.get('gender')
            if gender and gender.lower() in ['male', 'female']:
                query_parts.append(f'{gender.lower()}[MeSH Terms]')
            
            # Add age-related terms
            age_range = demographics.get('age_range')
            if age_range:
                min_age, max_age = age_range
                if min_age >= 65:
                    query_parts.append('aged[MeSH Terms]')
                elif min_age <= 18:
                    query_parts.append('(adolescent[MeSH Terms] OR child[MeSH Terms])')
                else:
                    query_parts.append('(adult[MeSH Terms] OR middle aged[MeSH Terms])')
        
        # Add publication date filter (last 10 years) with correct format
        query_parts.append('2014:2024[pdat]')
        
        # Add study type filters - use more general terms
        query_parts.append('(clinical trial[pt] OR comparative study[pt] OR evaluation studies[pt])')
        
        return ' AND '.join(query_parts)