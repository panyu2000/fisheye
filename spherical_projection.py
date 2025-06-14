import cv2
import numpy as np
import os
import time
from typing import Tuple, Dict, Optional
from camera_params import CameraParams

def apply_spherical_projection_maps(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
  """
  Apply spherical projection mapping to fisheye image using pre-generated maps.
  Now uses OpenCV's highly optimized remap function for 50-100x speedup.
  
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
  # This replaces the slow nested Python loops with optimized C++ implementation
  result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
  
  remap_time = time.time() - start_time
  print(f"OpenCV remap processing time: {remap_time:.4f} seconds")
  
  return result

class SphericalProjection:
  """
  Spherical projection processor for fisheye cameras with caching capabilities.
  
  This class represents a fisheye camera and provides efficient spherical projection
  functionality with automatic caching of projection maps for improved performance
  when processing multiple images with the same projection parameters.
  """
  
  def __init__(self, camera_params: CameraParams, input_image_size: Optional[Tuple[int, int]] = None, use_vectorized: bool = True):
    """
    Initialize SphericalProjection with camera parameters.
    
    Parameters:
    - camera_params: CameraParams object
    - input_image_size: (width, height) of input fisheye images, optional for validation
    - use_vectorized: if True, use fast vectorized map generation; if False, use reference implementation
    """
    self.camera_params = camera_params
    self.input_image_size = input_image_size
    self.use_vectorized = use_vectorized
    
    # Cache for projection maps - key is projection parameters tuple
    self._map_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
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
    """Generate a unique cache key for projection parameters."""
    behind_key = "behind_ok" if allow_behind_camera else "behind_skip"
    return f"{output_width}x{output_height}_yaw{yaw_offset:.3f}_pitch{pitch_offset:.3f}_fovh{fov_horizontal:.1f}_fovv{fov_vertical:.1f}_{behind_key}"
  
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
    print(f"Reference map generation processing time: {map_generation_time:.4f} seconds")
    
    return map_x, map_y
  
  def _generate_projection_maps_vectorized(self, output_width: int, output_height: int, 
                                          yaw_offset: float, pitch_offset: float, 
                                          fov_horizontal: float, fov_vertical: float, allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized implementation: Generate projection maps using NumPy array operations.
    
    This processes all pixels simultaneously for 10-50x speedup over the reference implementation.
    Uses the same mathematical logic but with vectorized operations.
    
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
    
    print(f"Generating spherical projection maps for camera: {output_width}x{output_height}")
    print(f"FOV: {fov_horizontal}° x {fov_vertical}°")
    print(f"Offsets: yaw={yaw_offset}°, pitch={pitch_offset}°")
    print(f"Allow behind camera: {allow_behind_camera}")
    
    # VECTORIZED IMPLEMENTATION: Process all pixels simultaneously
    # Create coordinate grids for all pixels
    u_coords, v_coords = np.meshgrid(np.arange(output_width), np.arange(output_height))
    
    # Convert ALL pixels to viewing direction angles simultaneously
    longitude = (u_coords / output_width - 0.5) * fov_h_rad + yaw_offset_rad
    latitude = (0.5 - v_coords / output_height) * fov_v_rad + pitch_offset_rad
    
    # Convert spherical angles to 3D unit vectors (vectorized)
    x_cam = np.cos(latitude) * np.sin(longitude)
    y_cam = np.sin(latitude)
    z_cam = np.cos(latitude) * np.cos(longitude)
    
    # Convert 3D vectors to fisheye projection angles (vectorized)
    theta = np.arccos(np.clip(z_cam, -1, 1))
    phi = np.arctan2(-y_cam, x_cam)
    
    # Create output arrays initialized to invalid coordinates
    map_x = np.full((output_height, output_width), -1.0, dtype=np.float32)
    map_y = np.full((output_height, output_width), -1.0, dtype=np.float32)
    
    # Create mask for valid pixels
    if allow_behind_camera:
      # All pixels are valid when behind camera is allowed
      valid_mask = np.ones((output_height, output_width), dtype=bool)
    else:
      # Only pixels within hemisphere (theta <= pi/2)
      valid_mask = theta <= np.pi/2
    
    # Process only valid pixels
    if np.any(valid_mask):
      theta_valid = theta[valid_mask]
      phi_valid = phi[valid_mask]
      
      # Apply fisheye distortion model (vectorized)
      theta_d = theta_valid * (1 + self.k1*theta_valid**2 + self.k2*theta_valid**4 + 
                              self.k3*theta_valid**6 + self.k4*theta_valid**8)
      
      # Convert to fisheye image coordinates (vectorized)
      x_fish = self.fx * theta_d * np.cos(phi_valid) + self.cx
      y_fish = self.fy * theta_d * np.sin(phi_valid) + self.cy
      
      # Store coordinates in maps
      map_x[valid_mask] = x_fish
      map_y[valid_mask] = y_fish
    
    map_generation_time = time.time() - start_time
    print(f"Vectorized map generation processing time: {map_generation_time:.4f} seconds")
    
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
    
    if cache_key in self._map_cache:
      print(f"Using cached projection maps: {cache_key}")
      return self._map_cache[cache_key]
    
    # Generate new maps
    map_x, map_y = self._generate_projection_maps(output_width, output_height, yaw_offset, pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    
    # Cache the maps
    self._map_cache[cache_key] = (map_x, map_y)
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
    # Validate input image size if specified
    if self.input_image_size is not None:
      img_height, img_width = input_img.shape[:2]
      expected_width, expected_height = self.input_image_size
      if img_width != expected_width or img_height != expected_height:
        raise ValueError(f"Input image size {img_width}x{img_height} does not match expected size {expected_width}x{expected_height}")
    
    # Get projection maps (cached or generate new)
    map_x, map_y = self.get_projection_maps(output_width, output_height, yaw_offset, pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    
    # Apply projection maps
    return apply_spherical_projection_maps(input_img, map_x, map_y)
  
  def clear_cache(self):
    """Clear all cached projection maps."""
    self._map_cache.clear()
    print("Projection map cache cleared")
  
  def get_cache_info(self) -> Dict[str, int]:
    """
    Get information about the current cache state.
    
    Returns:
    - Dictionary with cache statistics
    """
    total_memory = 0
    for map_x, map_y in self._map_cache.values():
      total_memory += map_x.nbytes + map_y.nbytes
    
    return {
      'cached_projections': len(self._map_cache),
      'memory_usage_bytes': total_memory,
      'memory_usage_mb': total_memory / (1024 * 1024)
    }
  
  def remove_cached_projection(self, output_width: int, output_height: int, 
                              yaw_offset: float, pitch_offset: float, 
                              fov_horizontal: float, fov_vertical: float, allow_behind_camera: bool = False):
    """Remove a specific projection from cache."""
    cache_key = self._generate_cache_key(output_width, output_height, yaw_offset, pitch_offset, fov_horizontal, fov_vertical, allow_behind_camera)
    if cache_key in self._map_cache:
      del self._map_cache[cache_key]
      print(f"Removed cached projection: {cache_key}")
    else:
      print(f"Projection not in cache: {cache_key}")
