"""
MIT License

Copyright (c) 2025 Pan Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Pan Yu
"""

import numpy as np
import threading
import time
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Any

class CacheManager:
  """
  Thread-safe LRU cache manager for projection maps.
  
  This class provides a centralized caching system for both perspective and spherical
  projection maps. It handles storage, retrieval, memory tracking, and cache management
  operations in a thread-safe manner with LRU (Least Recently Used) eviction policy.
  """
  
  def __init__(self, max_memory_mb: Optional[float] = None):
    """
    Initialize the cache manager with LRU eviction strategy.
    
    Parameters:
    - max_memory_mb: Optional maximum memory usage in MB. If None, no limit is enforced.
    """
    # Use OrderedDict to maintain insertion/access order for LRU
    self._cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray, float]] = OrderedDict()
    self._max_memory_mb = max_memory_mb
    self._lock = threading.RLock()  # Reentrant lock for thread safety
    self._access_count = 0  # Track total cache accesses for statistics
    self._eviction_count = 0  # Track number of evictions
  
  def get(self, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Retrieve cached projection maps with LRU update.
    
    Parameters:
    - cache_key: unique identifier for the cached projection maps
    
    Returns:
    - Tuple of (map_x, map_y) if found, None otherwise
    """
    with self._lock:
      self._access_count += 1
      
      if cache_key in self._cache:
        # Move to end (most recently used) and update timestamp
        map_x, map_y, _ = self._cache[cache_key]
        self._cache[cache_key] = (map_x, map_y, time.time())
        self._cache.move_to_end(cache_key)
        return (map_x, map_y)
      
      return None
  
  def put(self, cache_key: str, map_x: np.ndarray, map_y: np.ndarray) -> None:
    """
    Store projection maps in cache with LRU eviction when needed.
    
    Parameters:
    - cache_key: unique identifier for the projection maps
    - map_x: x-coordinate mapping array
    - map_y: y-coordinate mapping array
    """
    with self._lock:
      # Calculate memory for new entry
      new_memory_mb = (map_x.nbytes + map_y.nbytes) / (1024 * 1024)
      current_time = time.time()
      
      # If key already exists, update it
      if cache_key in self._cache:
        self._cache[cache_key] = (map_x.copy(), map_y.copy(), current_time)
        self._cache.move_to_end(cache_key)
        return
      
      # Check memory limit and evict if necessary
      if self._max_memory_mb is not None:
        current_memory = self._calculate_total_memory_mb()
        
        # Evict least recently used entries until we have enough space
        while (current_memory + new_memory_mb > self._max_memory_mb and 
               len(self._cache) > 0):
          
          # Remove least recently used item (first item in OrderedDict)
          lru_key, (lru_map_x, lru_map_y, _) = self._cache.popitem(last=False)
          freed_memory = (lru_map_x.nbytes + lru_map_y.nbytes) / (1024 * 1024)
          current_memory -= freed_memory
          self._eviction_count += 1
          
          print(f"LRU evicted: {lru_key} (freed {freed_memory:.1f} MB)")
        
        # If still not enough space after evicting all entries, warn and skip
        if current_memory + new_memory_mb > self._max_memory_mb:
          print(f"Warning: Cannot add cache entry - exceeds memory limit even after eviction "
                f"({self._max_memory_mb:.1f} MB)")
          return
      
      # Add new entry (will be placed at end as most recently used)
      self._cache[cache_key] = (map_x.copy(), map_y.copy(), current_time)
  
  def remove(self, cache_key: str) -> bool:
    """
    Remove a specific cache entry.
    
    Parameters:
    - cache_key: unique identifier for the projection maps to remove
    
    Returns:
    - True if the entry was found and removed, False otherwise
    """
    with self._lock:
      if cache_key in self._cache:
        del self._cache[cache_key]
        return True
      return False
  
  def clear(self) -> None:
    """Clear all cached projection maps."""
    with self._lock:
      self._cache.clear()
  
  def contains(self, cache_key: str) -> bool:
    """
    Check if a cache key exists.
    
    Parameters:
    - cache_key: unique identifier to check
    
    Returns:
    - True if the key exists in cache, False otherwise
    """
    with self._lock:
      return cache_key in self._cache
  
  def get_info(self) -> Dict[str, Any]:
    """
    Get comprehensive cache statistics including LRU metrics.
    
    Returns:
    - Dictionary with cache statistics including memory usage, entry counts, and LRU stats
    """
    with self._lock:
      total_memory_bytes = 0
      perspective_count = 0
      spherical_count = 0
      oldest_timestamp = float('inf')
      newest_timestamp = 0
      
      for key, (map_x, map_y, timestamp) in self._cache.items():
        total_memory_bytes += map_x.nbytes + map_y.nbytes
        oldest_timestamp = min(oldest_timestamp, timestamp)
        newest_timestamp = max(newest_timestamp, timestamp)
        
        if key.startswith('perspective_'):
          perspective_count += 1
        elif key.startswith('spherical_'):
          spherical_count += 1
      
      # Calculate cache age span
      cache_age_span = newest_timestamp - oldest_timestamp if len(self._cache) > 1 else 0
      
      return {
        'total_cached_projections': len(self._cache),
        'perspective_projections': perspective_count,
        'spherical_projections': spherical_count,
        'memory_usage_bytes': total_memory_bytes,
        'memory_usage_mb': total_memory_bytes / (1024 * 1024),
        'max_memory_mb': self._max_memory_mb,
        'memory_limit_enabled': self._max_memory_mb is not None,
        'total_accesses': self._access_count,
        'total_evictions': self._eviction_count,
        'cache_age_span_seconds': cache_age_span,
        'lru_enabled': True
      }
  
  def print_status(self) -> None:
    """Print current cache status in a human-readable format."""
    info = self.get_info()
    print(f"Cache status: {info['total_cached_projections']} projections "
          f"({info['perspective_projections']} perspective, {info['spherical_projections']} spherical), "
          f"{info['memory_usage_mb']:.1f} MB")
    
    if info['memory_limit_enabled']:
      usage_percent = (info['memory_usage_mb'] / info['max_memory_mb']) * 100
      print(f"Cache memory usage: {usage_percent:.1f}% of {info['max_memory_mb']:.1f} MB limit")
  
  def _calculate_total_memory_mb(self) -> float:
    """Calculate total memory usage in MB (internal method)."""
    total_bytes = 0
    for map_x, map_y, _ in self._cache.values():
      total_bytes += map_x.nbytes + map_y.nbytes
    return total_bytes / (1024 * 1024)
  
  def get_cache_keys(self, prefix: Optional[str] = None) -> list:
    """
    Get all cache keys, optionally filtered by prefix.
    
    Parameters:
    - prefix: Optional prefix to filter keys (e.g., 'perspective_' or 'spherical_')
    
    Returns:
    - List of cache keys
    """
    with self._lock:
      if prefix is None:
        return list(self._cache.keys())
      else:
        return [key for key in self._cache.keys() if key.startswith(prefix)]
  
  def get_memory_usage_by_type(self) -> Dict[str, float]:
    """
    Get memory usage breakdown by projection type.
    
    Returns:
    - Dictionary with memory usage in MB for each projection type
    """
    with self._lock:
      perspective_memory = 0
      spherical_memory = 0
      other_memory = 0
      
      for key, (map_x, map_y, _) in self._cache.items():
        memory_mb = (map_x.nbytes + map_y.nbytes) / (1024 * 1024)
        
        if key.startswith('perspective_'):
          perspective_memory += memory_mb
        elif key.startswith('spherical_'):
          spherical_memory += memory_mb
        else:
          other_memory += memory_mb
      
      return {
        'perspective_mb': perspective_memory,
        'spherical_mb': spherical_memory,
        'other_mb': other_memory,
        'total_mb': perspective_memory + spherical_memory + other_memory
      }
  
  def get_lru_order(self) -> list:
    """
    Get cache keys in LRU order (least recently used first).
    
    Returns:
    - List of cache keys ordered from least to most recently used
    """
    with self._lock:
      return list(self._cache.keys())
  
  def get_cache_ages(self) -> Dict[str, float]:
    """
    Get the age (in seconds) of each cache entry since last access.
    
    Returns:
    - Dictionary mapping cache keys to their ages in seconds
    """
    with self._lock:
      current_time = time.time()
      return {key: current_time - timestamp 
              for key, (_, _, timestamp) in self._cache.items()}
