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

import cv2
import numpy as np
import os
import time
from typing import Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from .camera_params import CameraParams
from .cache_manager import CacheManager

def apply_spherical_projection_maps(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
  """
  Apply spherical projection mapping to fisheye image using pre-generated maps.
  
  Parameters:
  - img: fisheye image as numpy array
  - map_x: array of x coordinates in fisheye image for each output pixel
  - map_y: array of y coordinates in fisheye image for each output pixel
  
  Returns:
  - spherical_img: projected spherical panorama image
  """
  if img is None:
    raise ValueError("Input image is None")
  
  output_height, output_width = map_x.shape
  
  print(f"Applying spherical projection maps using OpenCV remap to create {output_width}x{output_height} image")
  
  # Time the remap operation
  start_time = time.time()
  
  # Use OpenCV's highly optimized remap function with bilinear interpolation
  result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
  
  remap_time = time.time() - start_time
  print(f"\033[33mOpenCV remap processing time: {remap_time:.4f} seconds\033[0m")
  
  return result

class SphericalProjection:
  """
  Spherical projection processor for fisheye cameras with caching capabilities.
  
  This class represents a fisheye camera and provides efficient spherical projection
  functionality with automatic caching of projection maps for improved performance
  when processing multiple images with the same projection parameters.
  """
  
  def __init__(self, camera_params: CameraParams, use_vectorized: bool = True, cache_manager: Optional[CacheManager] = None):
    """
    Initialize SphericalProjection with camera parameters.
    
    Parameters:
    - camera_params: CameraParams object
    - use_vectorized: if True, use fast vectorized map generation; if False, use reference implementation
    - cache_manager: Optional shared cache manager. If None, creates a new one.
    """
    self.camera_params = camera_params
    self.use_vectorized = use_vectorized
    
    # Use shared cache manager or create a new one
    self.cache_manager = cache_manager if cache_manager is not None else CacheManager()
    
    # Extract camera parameters for easy access
    self.fx = camera_params.fx
    self.fy = camera_params.fy
    self.cx = camera_params.cx
    self.cy = camera_params.cy
    self.k1 = camera_params.k1
    self.k2 = camera_params.k2
    self.k3 = camera_params.k3
    self.k4 = camera_params.k4
  
  def _generate_cache_key(self, output_width: int, output_height: int, 
                         yaw_offset: float, pitch_offset: float, 
                         fov_horizontal: float, fov_vertical: float, allow_behind_camera: bool) -> str:
    """Generate a unique cache key for projection parameters with spherical prefix."""
    behind_key = "behind_ok" if allow_behind_camera else "behind_skip"
    return f"spherical_{output_width}x{output_height}_yaw{yaw_offset:.3f}_pitch{pitch_offset:.3f}_fovh{fov_horizontal:.1f}_fovv{fov_vertical:.1f}_{behind_key}"
  
  def _generate_projection_maps(self, output_width: int, output_height: int, 
                               yaw_offset: float, pitch_offset: float, 
                               fov_horizontal: float, fov_vertical: float, allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate projection maps using the camera's parameters.
    
    Dispatches to either vectorized (fast) or reference (slow but educational) implementation.
    
    Parameters:
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    
    Returns:
    - map_x, map_y: projection mapping arrays
    """
    if self.use_vectorized:
      print("Using vectorized (fast) map generation")
      return self._generate_projection_maps_vectorized(output_width, output_height, yaw_offset, 
                                                       pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    else:
      print("Using reference (slow but educational) map generation")
      return self._generate_projection_maps_reference(output_width, output_height, yaw_offset, 
                                                      pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
  
  def _generate_projection_maps_reference(self, output_width: int, output_height: int, 
                                         yaw_offset: float, pitch_offset: float, 
                                         fov_horizontal: float, fov_vertical: float, allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference implementation: Generate projection maps using nested loops.
    
    This is the original, slow but easy-to-understand implementation.
    Kept for educational purposes and debugging.
    
    Parameters:
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    """
    # Time the map generation process
    start_time = time.time()
    
    # Create mapping arrays
    map_x = np.full((output_height, output_width), -1.0, dtype=np.float32)
    map_y = np.full((output_height, output_width), -1.0, dtype=np.float32)
    
    # Convert offsets to radians
    yaw_offset_rad = np.radians(yaw_offset)
    pitch_offset_rad = np.radians(pitch_offset)
    
    # Calculate field of view in radians
    fov_h_rad = np.radians(fov_horizontal)
    fov_v_rad = np.radians(fov_vertical)
    
    print(f"Generating spherical projection maps for camera: {output_width}x{output_height}")
    print(f"FOV: {fov_horizontal}° x {fov_vertical}°")
    print(f"Offsets: yaw={yaw_offset}°, pitch={pitch_offset}°")
    print(f"Allow behind camera: {allow_behind_camera}")
    
    # REFERENCE IMPLEMENTATION: Nested loops (slow but educational)
    for v in range(output_height):
      for u in range(output_width):
        # Convert output pixel to viewing direction angles
        longitude = (u / output_width - 0.5) * fov_h_rad + yaw_offset_rad
        latitude = (0.5 - v / output_height) * fov_v_rad + pitch_offset_rad
        
        # Convert spherical angles to 3D unit vector
        x_cam = np.cos(latitude) * np.sin(longitude)
        y_cam = np.sin(latitude)
        z_cam = np.cos(latitude) * np.cos(longitude)
        
        # Convert 3D vector to fisheye projection angles
        theta = np.arccos(np.clip(z_cam, -1, 1))

        if not allow_behind_camera and theta > np.pi/2:
          # Skip if outside fisheye's hemisphere coverage
          continue

        phi = np.arctan2(-y_cam, x_cam)
        
        # Apply fisheye distortion model
        theta_d = theta * (1 + self.k1*theta**2 + self.k2*theta**4 + self.k3*theta**6 + self.k4*theta**8)
        
        # Convert to fisheye image coordinates
        x_fish = self.fx * theta_d * np.cos(phi) + self.cx
        y_fish = self.fy * theta_d * np.sin(phi) + self.cy
        
        # Store coordinates in maps
        map_x[v, u] = x_fish
        map_y[v, u] = y_fish
    
    map_generation_time = time.time() - start_time
    print(f"\033[33mReference map generation processing time: {map_generation_time:.4f} seconds\033[0m")
    
    return map_x, map_y
  
  def _process_row_chunk(self, row_start: int, row_end: int, output_width: int, output_height: int,
                        yaw_offset_rad: float, pitch_offset_rad: float, 
                        fov_h_rad: float, fov_v_rad: float, allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a chunk of rows in parallel for projection map generation.
    
    Parameters:
    - row_start, row_end: range of rows to process
    - output_width, output_height: dimensions of output image
    - yaw_offset_rad, pitch_offset_rad: rotation offsets in radians
    - fov_h_rad, fov_v_rad: field of view in radians
    - allow_behind_camera: if True, include rays pointing backward
    
    Returns:
    - Tuple of (map_x_chunk, map_y_chunk, valid_mask_chunk)
    """
    chunk_height = row_end - row_start
    
    # Create coordinate grids for this chunk
    u_coords, v_coords = np.meshgrid(
      np.arange(output_width, dtype=np.float32), 
      np.arange(row_start, row_end, dtype=np.float32)
    )
    
    # Convert pixels to viewing direction angles (optimized calculations)
    u_norm = u_coords / output_width - 0.5
    v_norm = 0.5 - v_coords / output_height
    
    longitude = u_norm * fov_h_rad + yaw_offset_rad
    latitude = v_norm * fov_v_rad + pitch_offset_rad
    
    # Pre-compute trigonometric values for better cache utilization
    cos_lat = np.cos(latitude)
    sin_lat = np.sin(latitude)
    cos_lon = np.cos(longitude)
    sin_lon = np.sin(longitude)
    
    # Convert spherical angles to 3D unit vectors (vectorized)
    x_cam = cos_lat * sin_lon
    y_cam = sin_lat
    z_cam = cos_lat * cos_lon
    
    # Convert 3D vectors to fisheye projection angles (vectorized)
    # Use more efficient clipping and avoid redundant calculations
    z_cam_clipped = np.clip(z_cam, -0.9999999, 0.9999999)  # Avoid numerical issues
    theta = np.arccos(z_cam_clipped)
    phi = np.arctan2(-y_cam, x_cam)
    
    # Create mask for valid pixels
    if allow_behind_camera:
      valid_mask = np.ones((chunk_height, output_width), dtype=bool)
    else:
      valid_mask = theta <= np.pi/2
    
    # Initialize output arrays for this chunk
    map_x_chunk = np.full((chunk_height, output_width), -1.0, dtype=np.float32)
    map_y_chunk = np.full((chunk_height, output_width), -1.0, dtype=np.float32)
    
    # Process only valid pixels if any exist
    if np.any(valid_mask):
      theta_valid = theta[valid_mask]
      phi_valid = phi[valid_mask]
      
      # Optimized fisheye distortion calculation using Horner's method
      theta2 = theta_valid * theta_valid
      theta4 = theta2 * theta2
      theta6 = theta4 * theta2
      theta8 = theta4 * theta4
      
      # Apply fisheye distortion model (optimized polynomial evaluation)
      distortion_factor = 1 + theta2 * (self.k1 + theta2 * (self.k2 + theta2 * (self.k3 + theta2 * self.k4)))
      theta_d = theta_valid * distortion_factor
      
      # Pre-compute trigonometric values for phi
      cos_phi = np.cos(phi_valid)
      sin_phi = np.sin(phi_valid)
      
      # Convert to fisheye image coordinates (vectorized)
      x_fish = self.fx * theta_d * cos_phi + self.cx
      y_fish = self.fy * theta_d * sin_phi + self.cy
      
      # Store coordinates in chunk maps
      map_x_chunk[valid_mask] = x_fish
      map_y_chunk[valid_mask] = y_fish
    
    return map_x_chunk, map_y_chunk, valid_mask

  def _generate_projection_maps_vectorized(self, output_width: int, output_height: int, 
                                          yaw_offset: float, pitch_offset: float, 
                                          fov_horizontal: float, fov_vertical: float, allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel vectorized implementation: Generate projection maps using NumPy array operations with multi-threading.
    
    This processes pixels in parallel chunks for optimal CPU utilization and cache efficiency.
    Provides significant speedup over the original vectorized implementation, especially for large images.
    
    Parameters:
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    """
    # Time the map generation process
    start_time = time.time()
    
    # Convert offsets to radians
    yaw_offset_rad = np.radians(yaw_offset)
    pitch_offset_rad = np.radians(pitch_offset)
    
    # Calculate field of view in radians
    fov_h_rad = np.radians(fov_horizontal)
    fov_v_rad = np.radians(fov_vertical)
    
    # Determine optimal number of threads and chunk size
    num_cores = min(multiprocessing.cpu_count(), 8)  # Cap at 8 threads to avoid overhead
    min_chunk_size = 32  # Minimum rows per chunk for cache efficiency
    chunk_size = max(min_chunk_size, output_height // (num_cores * 2))  # 2x cores for better load balancing
    
    print(f"Generating spherical projection maps for camera: {output_width}x{output_height}")
    print(f"FOV: {fov_horizontal}° x {fov_vertical}°")
    print(f"Offsets: yaw={yaw_offset}°, pitch={pitch_offset}°")
    print(f"Allow behind camera: {allow_behind_camera}")
    print(f"Using {num_cores} threads with chunk size {chunk_size} rows")
    
    # Create output arrays
    map_x = np.full((output_height, output_width), -1.0, dtype=np.float32)
    map_y = np.full((output_height, output_width), -1.0, dtype=np.float32)
    
    # For small images, use single-threaded processing to avoid overhead
    if output_height < 128 or output_width < 128:
      print("Using single-threaded processing for small image")
      map_x_chunk, map_y_chunk, _ = self._process_row_chunk(
        0, output_height, output_width, output_height,
        yaw_offset_rad, pitch_offset_rad, fov_h_rad, fov_v_rad, allow_behind_camera
      )
      map_x[:] = map_x_chunk
      map_y[:] = map_y_chunk
    else:
      # PARALLEL PROCESSING: Process image in chunks using ThreadPoolExecutor
      with ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Create tasks for each chunk
        futures = []
        row_ranges = []
        
        for row_start in range(0, output_height, chunk_size):
          row_end = min(row_start + chunk_size, output_height)
          row_ranges.append((row_start, row_end))
          
          future = executor.submit(
            self._process_row_chunk,
            row_start, row_end, output_width, output_height,
            yaw_offset_rad, pitch_offset_rad, fov_h_rad, fov_v_rad, allow_behind_camera
          )
          futures.append(future)
        
        # Collect results and assemble final maps
        for future, (row_start, row_end) in zip(futures, row_ranges):
          map_x_chunk, map_y_chunk, _ = future.result()
          map_x[row_start:row_end] = map_x_chunk
          map_y[row_start:row_end] = map_y_chunk
    
    map_generation_time = time.time() - start_time
    print(f"\033[33mParallel vectorized map generation processing time: {map_generation_time:.4f} seconds\033[0m")
    
    return map_x, map_y
  
  def get_projection_maps(self, output_width: int = 2048, output_height: int = 1024,
                         yaw_offset: float = 0, pitch_offset: float = 0, 
                         fov_horizontal: float = 360, fov_vertical: float = 180, allow_behind_camera: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get projection maps with caching.
    
    Returns cached maps if available, otherwise generates and caches new maps.
    
    Parameters:
    - output_width, output_height: dimensions of output panorama
    - yaw_offset, pitch_offset: rotation offsets in degrees
    - fov_horizontal, fov_vertical: field of view coverage in degrees
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    
    Returns:
    - map_x, map_y: projection mapping arrays
    """
    cache_key = self._generate_cache_key(output_width, output_height, yaw_offset, pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    
    # Try to get from cache
    cached_maps = self.cache_manager.get(cache_key)
    if cached_maps is not None:
      print(f"Using cached projection maps: {cache_key}")
      return cached_maps
    
    # Generate new maps
    map_x, map_y = self._generate_projection_maps(output_width, output_height, yaw_offset, pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    
    # Cache the maps
    self.cache_manager.put(cache_key, map_x, map_y)
    print(f"Cached projection maps: {cache_key}")
    
    return map_x, map_y
  
  def project(self, input_img: np.ndarray, output_width: int = 2048, output_height: int = 1024, 
             yaw_offset: float = 0, pitch_offset: float = 0, 
             fov_horizontal: float = 360, fov_vertical: float = 180, allow_behind_camera: bool = False) -> np.ndarray:
    """
    Apply spherical projection to input fisheye image.
    
    Parameters:
    - input_img: fisheye image as numpy array
    - output_width, output_height: dimensions of output panorama
    - yaw_offset, pitch_offset: rotation offsets in degrees
    - fov_horizontal, fov_vertical: field of view coverage in degrees
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    
    Returns:
    - spherical_img: projected spherical panorama image
    """
    # Validate input image size against camera parameters
    img_height, img_width = input_img.shape[:2]
    expected_width, expected_height = self.camera_params.width, self.camera_params.height
    if img_width != expected_width or img_height != expected_height:
      raise ValueError(f"Input image size {img_width}x{img_height} does not match camera parameters {expected_width}x{expected_height}")
    
    # Get projection maps (cached or generate new)
    map_x, map_y = self.get_projection_maps(output_width, output_height, yaw_offset, pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    
    # Print cache size after get_projection_maps is called
    self.cache_manager.print_status()
    
    # Apply projection maps
    return apply_spherical_projection_maps(input_img, map_x, map_y)
  
  def clear_cache(self):
    """Clear all cached projection maps."""
    self.cache_manager.clear()
    print("Projection map cache cleared")
  
  def get_cache_info(self) -> Dict[str, int]:
    """
    Get information about the current cache state.
    
    Returns:
    - Dictionary with cache statistics
    """
    return self.cache_manager.get_info()
  
  def remove_cached_projection(self, output_width: int, output_height: int, 
                              yaw_offset: float, pitch_offset: float, 
                              fov_horizontal: float, fov_vertical: float, allow_behind_camera: bool = False):
    """Remove a specific projection from cache."""
    cache_key = self._generate_cache_key(output_width, output_height, yaw_offset, pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    if self.cache_manager.remove(cache_key):
      print(f"Removed cached projection: {cache_key}")
    else:
      print(f"Projection not in cache: {cache_key}")
