import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from config.constants import COLLECTIONS, DEPRECATION_KEYWORDS
from retrieval.chroma_manager import ChromaManager

class DeprecationDetector:
    """Detects and handles deprecated APIs"""
    
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager
        self.deprecation_collection = COLLECTIONS["deprecations"]
    
    def check_deprecations(self, query: str, framework: str) -> Tuple[bool, Optional[Dict]]:
        """Check if query contains deprecated API references"""
        try:
            # First, search deprecation collection
            results = self.chroma_manager.query(
                collection_names=[self.deprecation_collection],
                query_text=query,
                n_results=5,
                where={"framework": framework} if framework else None
            )
            
            if not results:
                return False, None
            
            # Check if any result matches deprecation pattern
            for result in results:
                metadata = result.get('metadata', {})
                if metadata.get('is_deprecated'):
                    deprecation_info = {
                        'deprecated_api': metadata.get('source_title', 'Unknown'),
                        'replacement': metadata.get('replacement'),
                        'deprecation_notice': result.get('content', ''),
                        'version_deprecated': metadata.get('version'),
                        'source': metadata.get('source_url')
                    }
                    return True, deprecation_info
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking deprecations: {e}")
            return False, None
    
    def get_replacement_info(self, deprecated_api: str) -> List[Dict]:
        """Get information about replacement API"""
        try:
            # Search for replacement in API docs
            results = self.chroma_manager.query(
                collection_names=[COLLECTIONS["api_docs"]],
                query_text=deprecated_api,
                n_results=3
            )
            
            replacement_info = []
            for result in results:
                metadata = result.get('metadata', {})
                if not metadata.get('is_deprecated', True):
                    replacement_info.append({
                        'api_name': metadata.get('source_title'),
                        'documentation': result.get('content'),
                        'version': metadata.get('version'),
                        'source': metadata.get('source_url')
                    })
            
            return replacement_info
            
        except Exception as e:
            logger.error(f"Error getting replacement info: {e}")
            return []
    
    def extract_deprecated_apis(self, query: str) -> List[str]:
        """Extract potential deprecated API names from query"""
        # Common API patterns
        api_patterns = [
            r'(\w+\.\w+\(\))',  # module.function()
            r'(\w+\.\w+\.\w+\(\))',  # module.submodule.function()
            r'(\w+\(\))',  # function()
            r'(\w+\.\w+)'  # module.class
        ]
        
        deprecated_apis = []
        
        for pattern in api_patterns:
            matches = re.findall(pattern, query)
            deprecated_apis.extend(matches)
        
        return deprecated_apis
    
    def get_migration_path(self, from_version: str, to_version: str, framework: str) -> List[Dict]:
        """Get migration path between versions"""
        try:
            # Search release notes for migration information
            results = self.chroma_manager.query(
                collection_names=[COLLECTIONS["release_notes"]],
                query_text=f"migration from {from_version} to {to_version}",
                n_results=5,
                where={
                    "framework": framework,
                    "version": {
                        "$gte": from_version,
                        "$lte": to_version
                    }
                }
            )
            
            migration_steps = []
            for result in results:
                migration_steps.append({
                    'version': result.get('metadata', {}).get('version'),
                    'changes': result.get('content'),
                    'source': result.get('metadata', {}).get('source_url')
                })
            
            return migration_steps
            
        except Exception as e:
            logger.error(f"Error getting migration path: {e}")
            return []