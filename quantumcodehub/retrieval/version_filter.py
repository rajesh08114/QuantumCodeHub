from typing import Dict, Any, Optional, Tuple
from packaging import version
from loguru import logger

from config.constants import Framework, SUPPORTED_VERSIONS

class VersionFilter:
    """Handles version-aware filtering logic"""
    
    def __init__(self):
        self.supported_versions = SUPPORTED_VERSIONS
    
    def get_version_filter(self, 
                          framework: Framework, 
                          requested_version: Optional[str] = None) -> Dict[str, Any]:
        """Get version filter for ChromaDB query"""
        
        # Get supported range for framework
        min_version, max_version = self.supported_versions.get(
            framework, ("0.0.0", "999.9.9")
        )
        
        if requested_version:
            # Check if requested version is supported
            if self._is_version_supported(framework, requested_version):
                return {
                    "version": {
                        "$eq": requested_version
                    }
                }
            else:
                logger.warning(f"Requested version {requested_version} not supported for {framework.value}")
                # Fall back to supported range
                return self._get_range_filter(min_version, max_version)
        else:
            # No version requested, use supported range
            return self._get_range_filter(min_version, max_version)
    
    def _get_range_filter(self, min_version: str, max_version: str) -> Dict[str, Any]:
        """Create range filter for versions"""
        return {
            "version": {
                "$gte": min_version,
                "$lte": max_version
            }
        }
    
    def _is_version_supported(self, framework: Framework, version_str: str) -> bool:
        """Check if version is within supported range"""
        try:
            req_version = version.parse(version_str)
            min_ver, max_ver = self.supported_versions.get(
                framework, ("0.0.0", "999.9.9")
            )
            
            min_version = version.parse(min_ver)
            max_version = version.parse(max_ver)
            
            return min_version <= req_version <= max_version
            
        except Exception as e:
            logger.error(f"Error parsing version {version_str}: {e}")
            return False
    
    def calculate_version_match_score(self, 
                                     doc_version: str, 
                                     query_version: Optional[str],
                                     framework: Framework) -> float:
        """Calculate how well document version matches query"""
        try:
            doc_ver = version.parse(doc_version)
            
            if query_version:
                # Exact match requested
                query_ver = version.parse(query_version)
                if doc_ver == query_ver:
                    return 1.0
                
                # Check if close version
                if abs(doc_ver.major - query_ver.major) == 0:
                    if abs(doc_ver.minor - query_ver.minor) <= 1:
                        return 0.8
                
                return 0.3
            else:
                # No version requested, score based on how recent
                min_ver, max_ver = self.supported_versions.get(
                    framework, ("0.0.0", "999.9.9")
                )
                
                # Prefer latest version
                max_version = version.parse(max_ver)
                if doc_ver == max_version:
                    return 1.0
                
                # Score based on proximity to latest
                if doc_ver.major == max_version.major:
                    diff = max_version.minor - doc_ver.minor
                    return max(0.5, 1.0 - (diff * 0.1))
                
                return 0.3
                
        except Exception as e:
            logger.error(f"Error calculating version match score: {e}")
            return 0.5