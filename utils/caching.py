"""
Cache Management Utilities
Query result caching with TTL and size limits
"""

import re
from datetime import datetime
from typing import List, Dict, Optional

# Import from config
from config import MAX_CACHE_SIZE, CACHE_TTL_SECONDS

class QueryCache:
    """
    Restaurant-aware query cache with TTL and size limits
    
    Uses (normalized_query, restaurant_name) as cache key for precision
    """
    
    def __init__(self, max_size=MAX_CACHE_SIZE, ttl_seconds=CACHE_TTL_SECONDS):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cached results in seconds
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for cache key
        
        Args:
            query: Search query
        
        Returns:
            Normalized query string
        """
        # Lowercase and strip
        normalized = query.lower().strip()
        
        # Remove punctuation
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _make_key(self, query: str, restaurant: Optional[str] = None) -> tuple:
        """
        Create cache key from query and optional restaurant
        
        Args:
            query: Search query
            restaurant: Restaurant name (optional)
        
        Returns:
            Tuple cache key
        """
        normalized = self._normalize_query(query)
        return (normalized, restaurant if restaurant else "")
    
    def get(self, query: str, restaurant: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get cached results if available and fresh
        
        Args:
            query: Search query
            restaurant: Restaurant name (optional)
        
        Returns:
            Cached results or None
        """
        cache_key = self._make_key(query, restaurant)
        
        if cache_key not in self.cache:
            return None
        
        cached_query, results, timestamp = self.cache[cache_key]
        
        # Check if fresh
        age = (datetime.now() - timestamp).total_seconds()
        if age < self.ttl_seconds:
            print(f"Cache HIT (exact match)! Query: '{cached_query}'")
            return results
        else:
            print(f"Cache EXPIRED: {age/60:.1f} minutes old")
            del self.cache[cache_key]
            return None
    
    def set(self, query: str, results: List[Dict], restaurant: Optional[str] = None):
        """
        Cache query results
        
        Args:
            query: Search query
            results: Search results to cache
            restaurant: Restaurant name (optional)
        """
        cache_key = self._make_key(query, restaurant)
        
        self.cache[cache_key] = (query, results, datetime.now())
        
        # Limit cache size (LRU eviction)
        if len(self.cache) > self.max_size:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][2]  # Sort by timestamp
            )
            del self.cache[oldest_key]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
    
    def stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        if not self.cache:
            return {
                'size': 0,
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'oldest_age_seconds': None,
                'newest_age_seconds': None
            }
        
        now = datetime.now()
        ages = [(now - timestamp).total_seconds() for _, _, timestamp in self.cache.values()]
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'oldest_age_seconds': max(ages),
            'newest_age_seconds': min(ages)
        }