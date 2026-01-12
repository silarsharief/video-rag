"""
Caching System for Query Rewrites and Search Results
Reduces API calls and improves response times.
"""
import time
import logging
from collections import OrderedDict
from typing import Optional, Any
import hashlib

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Least Recently Used (LRU) cache with TTL support.
    
    Features:
    - LRU eviction when max size reached
    - Time-to-live (TTL) expiration
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize LRU cache.
        
        Args:
            max_size (int): Maximum number of entries (default: 100)
            ttl (int): Time-to-live in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # {key: (value, timestamp)}
        self.hits = 0
        self.misses = 0
        
        logger.info(f"🗄️  Cache initialized: max_size={max_size}, ttl={ttl}s")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: Hash-based cache key
        """
        # Create a string representation of all arguments
        key_str = str(args) + str(sorted(kwargs.items()))
        # Generate hash for compact key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if entry is expired based on TTL."""
        return (time.time() - timestamp) > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Any: Cached value if found and not expired, None otherwise
        """
        if key not in self.cache:
            self.misses += 1
            logger.debug(f"❌ Cache miss: {key[:8]}...")
            return None
        
        value, timestamp = self.cache[key]
        
        # Check if expired
        if self._is_expired(timestamp):
            logger.debug(f"⏰ Cache expired: {key[:8]}...")
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        logger.debug(f"✅ Cache hit: {key[:8]}...")
        return value
    
    def put(self, key: str, value: Any):
        """
        Put value into cache.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
        """
        # Update existing entry
        if key in self.cache:
            self.cache.move_to_end(key)
        
        # Add new entry
        self.cache[key] = (value, time.time())
        logger.debug(f"💾 Cache put: {key[:8]}...")
        
        # Evict oldest if over capacity
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"🗑️  Cache evicted (LRU): {oldest_key[:8]}...")
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("🗑️  Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_accesses = self.hits + self.misses
        hit_rate = (self.hits / total_accesses * 100) if total_accesses > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl': self.ttl
        }
    
    def __len__(self):
        """Get number of entries in cache."""
        return len(self.cache)
    
    def __contains__(self, key):
        """Check if key exists in cache."""
        if key not in self.cache:
            return False
        value, timestamp = self.cache[key]
        return not self._is_expired(timestamp)


class QueryCache:
    """
    Specialized cache for query rewrites.
    
    Caches rewritten queries to avoid repeated API calls for same queries.
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize query cache.
        
        Args:
            max_size (int): Maximum number of cached queries
            ttl (int): Time-to-live in seconds
        """
        self.cache = LRUCache(max_size=max_size, ttl=ttl)
        logger.info("🔍 Query cache initialized")
    
    def get_rewritten_query(self, original_query: str, mode: str) -> Optional[str]:
        """
        Get cached rewritten query.
        
        Args:
            original_query (str): Original user query
            mode (str): Query mode (traffic, factory, etc.)
            
        Returns:
            str: Cached rewritten query if found, None otherwise
        """
        cache_key = self.cache._generate_key(original_query, mode)
        result = self.cache.get(cache_key)
        
        if result is not None:
            logger.info(f"💾 Using cached query rewrite for: '{original_query[:50]}...'")
        
        return result
    
    def put_rewritten_query(self, original_query: str, mode: str, rewritten_query: str):
        """
        Cache a rewritten query.
        
        Args:
            original_query (str): Original user query
            mode (str): Query mode
            rewritten_query (str): Rewritten enhanced query
        """
        cache_key = self.cache._generate_key(original_query, mode)
        self.cache.put(cache_key, rewritten_query)
        logger.debug(f"💾 Cached query rewrite: '{original_query[:30]}...' -> '{rewritten_query[:30]}...'")
    
    def clear(self):
        """Clear query cache."""
        self.cache.clear()
        logger.info("🗑️  Query cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()


# Global cache instances (singletons)
_query_cache = None


def get_query_cache() -> QueryCache:
    """Get the global query cache instance."""
    global _query_cache
    if _query_cache is None:
        from config.settings import QUERY_CACHE_MAX_SIZE, QUERY_CACHE_TTL
        _query_cache = QueryCache(max_size=QUERY_CACHE_MAX_SIZE, ttl=QUERY_CACHE_TTL)
    return _query_cache


def reset_caches():
    """Reset all global caches."""
    global _query_cache
    if _query_cache is not None:
        _query_cache.clear()

