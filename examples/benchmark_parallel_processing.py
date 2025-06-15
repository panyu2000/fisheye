"""
Benchmark script to demonstrate the performance improvements from parallel processing optimizations.

This script compares the performance of projection map generation with different image sizes
to show the benefits of the new parallel processing implementation.
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera_params import parse_camera_params
from src.perspective_projection import PerspectiveProjection
from src.spherical_projection import SphericalProjection

def benchmark_projection_performance():
  """Benchmark the performance improvements from parallel processing."""
  
  print("=" * 60)
  print("FISHEYE PROJECTION PARALLEL PROCESSING BENCHMARK")
  print("=" * 60)
  
  # Load camera parameters
  try:
    camera_params = parse_camera_params("config/camera_intrinsics.yaml")
    print(f"✓ Loaded camera parameters: {camera_params.width}x{camera_params.height}")
  except Exception as e:
    print(f"✗ Error loading camera parameters: {e}")
    return
  
  # Create projection instances
  perspective_projector = PerspectiveProjection(camera_params, use_vectorized=True)
  spherical_projector = SphericalProjection(camera_params, use_vectorized=True)
  
  # Test different output sizes to show scaling benefits
  test_sizes = [
    (512, 512, "Small"),
    (1024, 1024, "Medium"),
    (2048, 1024, "Large"),
    (4096, 2048, "Very Large")
  ]
  
  print("\n" + "=" * 60)
  print("PERSPECTIVE PROJECTION BENCHMARKS")
  print("=" * 60)
  
  for width, height, size_name in test_sizes:
    print(f"\n{size_name} image size: {width}x{height}")
    print("-" * 40)
    
    try:
      # Benchmark perspective projection
      start_time = time.time()
      map_x, map_y = perspective_projector.get_projection_maps(
        output_width=width,
        output_height=height,
        yaw_offset=15.0,
        pitch_offset=10.0,
        roll_offset=5.0,
        fov_horizontal=90.0
      )
      total_time = time.time() - start_time
      
      # Calculate pixels per second
      total_pixels = width * height
      pixels_per_second = total_pixels / total_time if total_time > 0 else 0
      
      print(f"✓ Total processing time: {total_time:.4f} seconds")
      print(f"✓ Pixels processed: {total_pixels:,}")
      print(f"✓ Performance: {pixels_per_second:,.0f} pixels/second")
      print(f"✓ Memory usage: {(map_x.nbytes + map_y.nbytes) / 1024 / 1024:.1f} MB")
      
    except Exception as e:
      print(f"✗ Error processing {size_name}: {e}")
  
  print("\n" + "=" * 60)
  print("SPHERICAL PROJECTION BENCHMARKS")
  print("=" * 60)
  
  for width, height, size_name in test_sizes:
    print(f"\n{size_name} image size: {width}x{height}")
    print("-" * 40)
    
    try:
      # Benchmark spherical projection
      start_time = time.time()
      map_x, map_y = spherical_projector.get_projection_maps(
        output_width=width,
        output_height=height,
        yaw_offset=20.0,
        pitch_offset=0.0,
        fov_horizontal=360.0,
        fov_vertical=180.0
      )
      total_time = time.time() - start_time
      
      # Calculate pixels per second
      total_pixels = width * height
      pixels_per_second = total_pixels / total_time if total_time > 0 else 0
      
      print(f"✓ Total processing time: {total_time:.4f} seconds")
      print(f"✓ Pixels processed: {total_pixels:,}")
      print(f"✓ Performance: {pixels_per_second:,.0f} pixels/second")
      print(f"✓ Memory usage: {(map_x.nbytes + map_y.nbytes) / 1024 / 1024:.1f} MB")
      
    except Exception as e:
      print(f"✗ Error processing {size_name}: {e}")
  
  # Display cache information
  print("\n" + "=" * 60)
  print("CACHE PERFORMANCE")
  print("=" * 60)
  
  perspective_cache_info = perspective_projector.get_cache_info()
  spherical_cache_info = spherical_projector.get_cache_info()
  
  print(f"Perspective cache:")
  print(f"  ✓ Cached projections: {perspective_cache_info['cached_projections']}")
  print(f"  ✓ Memory usage: {perspective_cache_info['memory_usage_mb']:.1f} MB")
  
  print(f"Spherical cache:")
  print(f"  ✓ Cached projections: {spherical_cache_info['cached_projections']}")
  print(f"  ✓ Memory usage: {spherical_cache_info['memory_usage_mb']:.1f} MB")
  
  # Test cache hit performance
  print(f"\nTesting cache hit performance...")
  print("-" * 40)
  
  # Test perspective cache hit
  start_time = time.time()
  map_x, map_y = perspective_projector.get_projection_maps(
    output_width=1024, output_height=1024,
    yaw_offset=15.0, pitch_offset=10.0, roll_offset=5.0, fov_horizontal=90.0
  )
  cache_hit_time = time.time() - start_time
  print(f"✓ Perspective cache hit time: {cache_hit_time:.6f} seconds")
  
  # Test spherical cache hit
  start_time = time.time()
  map_x, map_y = spherical_projector.get_projection_maps(
    output_width=1024, output_height=1024,
    yaw_offset=20.0, pitch_offset=0.0, fov_horizontal=360.0, fov_vertical=180.0
  )
  cache_hit_time = time.time() - start_time
  print(f"✓ Spherical cache hit time: {cache_hit_time:.6f} seconds")
  
  print("\n" + "=" * 60)
  print("OPTIMIZATION SUMMARY")
  print("=" * 60)
  print("✓ Implemented parallel processing with multi-threading")
  print("✓ Optimized mathematical operations using Horner's method")
  print("✓ Improved cache efficiency with chunked processing")
  print("✓ Enhanced memory access patterns for better performance")
  print("✓ Added automatic thread count optimization based on CPU cores")
  print("✓ Included smart fallback to single-threading for small images")
  print("=" * 60)

if __name__ == "__main__":
  benchmark_projection_performance()
