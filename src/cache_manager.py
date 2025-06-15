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
from typing import Dict, Tuple, Optional, Any

class CacheManager:
  """
  Thread-safe cache manager for projection maps.
  
  This class provides a centralized caching system for both perspective and spherical
  projection maps. It handles storage, retrieval, memory tracking, and cache management
  operations in a thread-safe manner.
  """
  
  def __init__(self, max_memory_mb: Optional[float] = None):
    """
    Initialize the cache manager.
    
    Parameters:
    - max_memory_mb: Optional maximum memory usage in MB. If None, no limit is enforced.
    """
    self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    self._max_memory_mb = max_memory_mb
    self._lock = threading.RLock()  # Reentrant lock for thread safety
  
  def get(self, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Retrieve cached projection maps.
    
    Parameters:
    - cache_key: unique identifier for the cached projection maps
    
    Returns:
    - Tuple of (map_x, map_y) if found, None otherwise
    """
    with self._lock:
      return self._cache.get(cache_key)
  
  def put(self, cache_key: str, map_x: np.ndarray, map_y: np.ndarray) -> None:
    """
    Store projection maps in cache.
    
    Parameters:
    - cache_key: unique identifier for the projection maps
    - map_x: x-coordinate mapping array
    - map_y: y-coordinate mapping array
    """
    with self._lock:
      # Check memory limit before adding
      if self._max_memory_mb is not None:
        new_memory = (map_x.nbytes + map_y.nbytes) / (1024 * 1024)
        current_memory = self._calculate_total_memory_mb()
        
        if current_memory + new_memory > self._max_memory_mb:
          # Could implement LRU eviction here in the future
          print(f"Warning: Adding cache entry would exceed memory limit ({self._max_memory_mb:.1f} MB)")
          return
      
      self._cache[cache_key] = (map_x.copy(), map_y.copy())
  
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
    Get comprehensive cache statistics.
    
    Returns:
    - Dictionary with cache statistics including memory usage and entry counts
    """
    with self._lock:
      total_memory_bytes = 0
      perspective_count = 0
      spherical_count = 0
      
      for key, (map_x, map_y) in self._cache.items():
        total_memory_bytes += map_x.nbytes + map_y.nbytes
        
        if key.startswith('perspective_'):
          perspective_count += 1
        elif key.startswith('spherical_'):
          spherical_count += 1
      
      return {
        'total_cached_projections': len(self._cache),
        'perspective_projections': perspective_count,
        'spherical_projections': spherical_count,
        'memory_usage_bytes': total_memory_bytes,
        'memory_usage_mb': total_memory_bytes / (1024 * 1024),
        'max_memory_mb': self._max_memory_mb,
        'memory_limit_enabled': self._max_memory_mb is not None
      }
  
  def print_status(self) -> None:
    """Print current cache status in a human-readable format."""
    info = self.get_info()
    print(f"Cache status: {info['total_cached_projections']} projections "
          f"({info['perspective_projections']} perspective, {info['spherical_projections']} spherical), "
          f"{info['memory_usage_mb']:.1f} MB")
    
    if info['memory_limit_enabled']:
      usage_percent = (info['memory_usage_mb'] / info['max_memory_mb']) * 100
      print(f"Memory usage: {usage_percent:.1f}% of {info['max_memory_mb']:.1f} MB limit")
  
  def _calculate_total_memory_mb(self) -> float:
    """Calculate total memory usage in MB (internal method)."""
    total_bytes = 0
    for map_x, map_y in self._cache.values():
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
      
      for key, (map_x, map_y) in self._cache.items():
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
