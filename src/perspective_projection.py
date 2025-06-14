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
from .camera_params import CameraParams

def apply_perspective_projection_maps(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
  """
  Apply perspective projection mapping to fisheye image using pre-generated maps.
  Now uses OpenCV's highly optimized remap function for 50-100x speedup.
  
  Parameters:
  - img: fisheye image as numpy array
  - map_x: array of x coordinates in fisheye image for each output pixel
  - map_y: array of y coordinates in fisheye image for each output pixel
  
  Returns:
  - perspective_img: projected perspective image
  """
  if img is None:
    raise ValueError("Input image is None")
  
  output_height, output_width = map_x.shape
  
  print(f"Applying perspective projection maps using OpenCV remap to create {output_width}x{output_height} image")
  
  # Time the remap operation
  start_time = time.time()
  
  # Use OpenCV's highly optimized remap function with bilinear interpolation
  # This replaces the slow nested Python loops with optimized C++ implementation
  result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
  
  remap_time = time.time() - start_time
  print(f"OpenCV remap processing time: {remap_time:.4f} seconds")
  
  return result

class PerspectiveProjection:
  """
  Perspective projection processor for fisheye cameras with caching capabilities.
  
  This class represents a fisheye camera and provides efficient perspective projection
  functionality with automatic caching of projection maps for improved performance
  when processing multiple images with the same projection parameters.
  """
  
  def __init__(self, camera_params: CameraParams, input_image_size: Optional[Tuple[int, int]] = None, use_vectorized: bool = True):
    """
    Initialize PerspectiveProjection with camera parameters.
    
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
                         yaw_offset: float, pitch_offset: float, roll_offset: float,
                         fov_horizontal: float, virtual_fx: Optional[float], virtual_fy: Optional[float],
                         allow_behind_camera: bool) -> str:
    """Generate a unique cache key for projection parameters."""
    fx_key = f"fx{virtual_fx:.1f}" if virtual_fx is not None else "fx_auto"
    fy_key = f"fy{virtual_fy:.1f}" if virtual_fy is not None else "fy_auto"
    behind_key = "behind_ok" if allow_behind_camera else "behind_skip"
    return f"{output_width}x{output_height}_yaw{yaw_offset:.3f}_pitch{pitch_offset:.3f}_roll{roll_offset:.3f}_fovh{fov_horizontal:.1f}_{fx_key}_{fy_key}_{behind_key}"
  
  def _generate_projection_maps(self, output_width: int, output_height: int, 
                               yaw_offset: float, pitch_offset: float, roll_offset: float,
                               fov_horizontal: float, virtual_fx: Optional[float], virtual_fy: Optional[float],
                               allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray]:
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
                                                       pitch_offset, roll_offset, fov_horizontal, 
                                                       virtual_fx, virtual_fy, allow_behind_camera)
    else:
      print("Using reference (slow but educational) map generation")
      return self._generate_projection_maps_reference(output_width, output_height, yaw_offset, 
                                                      pitch_offset, roll_offset, fov_horizontal, 
                                                      virtual_fx, virtual_fy, allow_behind_camera)
  
  def _generate_projection_maps_reference(self, output_width: int, output_height: int, 
                                         yaw_offset: float, pitch_offset: float, roll_offset: float,
                                         fov_horizontal: float, virtual_fx: Optional[float], virtual_fy: Optional[float],
                                         allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray]:
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
    
    # Convert rotation offsets to radians
    yaw_rad = np.radians(yaw_offset)
    pitch_rad = np.radians(pitch_offset)
    roll_rad = np.radians(roll_offset)
    
    # Calculate virtual camera parameters
    if virtual_fx is None:
      # Calculate focal length from horizontal FOV
      virtual_fx = (output_width / 2.0) / np.tan(np.radians(fov_horizontal) / 2.0)
    if virtual_fy is None:
      virtual_fy = virtual_fx  # Assume square pixels
    
    virtual_cx = output_width / 2.0
    virtual_cy = output_height / 2.0
    
    print(f"Generating perspective projection maps for camera: {output_width}x{output_height}")
    print(f"Virtual camera FOV: {fov_horizontal}°")
    print(f"Virtual camera params: fx={virtual_fx:.1f}, fy={virtual_fy:.1f}")
    print(f"Rotation: yaw={yaw_offset}°, pitch={pitch_offset}°, roll={roll_offset}°")
    print(f"Allow behind camera: {allow_behind_camera}")
    
    # Create rotation matrices using standard camera coordinate conventions
    # Yaw: rotation around Y-axis (left/right turn)
    R_yaw = np.array([
      [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
      [0, 1, 0],
      [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    # Pitch: rotation around X-axis (up/down tilt)
    # Positive pitch should tilt camera upwards
    R_pitch = np.array([
      [1, 0, 0],
      [0, np.cos(pitch_rad), np.sin(pitch_rad)],
      [0, -np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Roll: rotation around Z-axis (camera tilt)
    R_roll = np.array([
      [np.cos(roll_rad), -np.sin(roll_rad), 0],
      [np.sin(roll_rad), np.cos(roll_rad), 0],
      [0, 0, 1]
    ])
    
    # Combined rotation matrix (apply in order: roll, pitch, yaw)
    R_combined = R_yaw @ R_pitch @ R_roll
    
    # REFERENCE IMPLEMENTATION: Nested loops (slow but educational)
    for v in range(output_height):
      for u in range(output_width):
        # Convert output pixel to 3D ray direction in virtual camera coordinates
        x_virtual = (u - virtual_cx) / virtual_fx
        y_virtual = -(v - virtual_cy) / virtual_fy  # Negative to correct image orientation
        z_virtual = 1.0
        
        # Create 3D direction vector
        ray_direction = np.array([x_virtual, y_virtual, z_virtual])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize
        
        # Apply rotation to get direction in fisheye camera coordinate system
        rotated_direction = R_combined @ ray_direction
        
        x_cam, y_cam, z_cam = rotated_direction
        
        # Skip if ray points backward (negative z) and not allowed
        if not allow_behind_camera and z_cam <= 0:
          continue
        
        # Convert 3D direction to fisheye projection angles
        theta = np.arccos(np.clip(z_cam, -1, 1))
        
        # Skip if outside fisheye's hemisphere coverage (when not allowing behind camera)
        if not allow_behind_camera and theta > np.pi/2:
          continue
        
        # Calculate azimuth angle
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
                                          yaw_offset: float, pitch_offset: float, roll_offset: float,
                                          fov_horizontal: float, virtual_fx: Optional[float], virtual_fy: Optional[float],
                                          allow_behind_camera: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized implementation: Generate projection maps using NumPy array operations.
    
    This processes all pixels simultaneously for 10-50x speedup over the reference implementation.
    Uses the same mathematical logic but with vectorized operations.
    
    Parameters:
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    """
    # Time the map generation process
    start_time = time.time()
    
    # Convert rotation offsets to radians
    yaw_rad = np.radians(yaw_offset)
    pitch_rad = np.radians(pitch_offset)
    roll_rad = np.radians(roll_offset)
    
    # Calculate virtual camera parameters
    if virtual_fx is None:
      virtual_fx = (output_width / 2.0) / np.tan(np.radians(fov_horizontal) / 2.0)
    if virtual_fy is None:
      virtual_fy = virtual_fx
    
    virtual_cx = output_width / 2.0
    virtual_cy = output_height / 2.0
    
    print(f"Generating perspective projection maps for camera: {output_width}x{output_height}")
    print(f"Virtual camera FOV: {fov_horizontal}°")
    print(f"Virtual camera params: fx={virtual_fx:.1f}, fy={virtual_fy:.1f}")
    print(f"Rotation: yaw={yaw_offset}°, pitch={pitch_offset}°, roll={roll_offset}°")
    print(f"Allow behind camera: {allow_behind_camera}")
    
    # Create rotation matrices
    R_yaw = np.array([
      [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
      [0, 1, 0],
      [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    R_pitch = np.array([
      [1, 0, 0],
      [0, np.cos(pitch_rad), np.sin(pitch_rad)],
      [0, -np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
      [np.cos(roll_rad), -np.sin(roll_rad), 0],
      [np.sin(roll_rad), np.cos(roll_rad), 0],
      [0, 0, 1]
    ])
    
    R_combined = R_yaw @ R_pitch @ R_roll
    
    # VECTORIZED IMPLEMENTATION: Process all pixels simultaneously
    # Create coordinate grids for all pixels
    u_coords, v_coords = np.meshgrid(np.arange(output_width), np.arange(output_height))
    
    # Convert ALL pixels to virtual camera coordinates simultaneously
    x_virtual = (u_coords - virtual_cx) / virtual_fx
    y_virtual = -(v_coords - virtual_cy) / virtual_fy
    z_virtual = np.ones_like(x_virtual)
    
    # Stack coordinates for vectorized operations
    # Shape: (3, height, width)
    ray_directions = np.stack([x_virtual, y_virtual, z_virtual], axis=0)
    
    # Normalize all ray directions simultaneously
    norms = np.linalg.norm(ray_directions, axis=0)
    ray_directions = ray_directions / norms[np.newaxis, :, :]
    
    # Apply rotation to ALL rays simultaneously using Einstein summation
    # This is equivalent to matrix multiplication for each pixel
    rotated_directions = np.einsum('ij,jhw->ihw', R_combined, ray_directions)
    
    x_cam = rotated_directions[0]
    y_cam = rotated_directions[1]
    z_cam = rotated_directions[2]
    
    # Create output arrays initialized to invalid coordinates
    map_x = np.full((output_height, output_width), -1.0, dtype=np.float32)
    map_y = np.full((output_height, output_width), -1.0, dtype=np.float32)
    
    # Create mask for valid pixels
    if allow_behind_camera:
      # All pixels are valid when behind camera is allowed
      valid_mask = np.ones((output_height, output_width), dtype=bool)
    else:
      # Only forward-facing rays and within hemisphere
      valid_mask = z_cam > 0
    
    # Process only valid pixels
    if np.any(valid_mask):
      x_cam_valid = x_cam[valid_mask]
      y_cam_valid = y_cam[valid_mask]
      z_cam_valid = z_cam[valid_mask]
      
      # Convert 3D direction to fisheye projection angles (vectorized)
      theta = np.arccos(np.clip(z_cam_valid, -1, 1))
      
      # Additional hemisphere check when not allowing behind camera
      if not allow_behind_camera:
        hemisphere_mask = theta <= np.pi/2
        if not np.any(hemisphere_mask):
          map_generation_time = time.time() - start_time
          print(f"Vectorized map generation processing time: {map_generation_time:.4f} seconds")
          return map_x, map_y
        
        # Further filter valid pixels
        theta = theta[hemisphere_mask]
        x_cam_valid = x_cam_valid[hemisphere_mask]
        y_cam_valid = y_cam_valid[hemisphere_mask]
        
        # Update the valid mask to reflect hemisphere filtering
        temp_mask = np.zeros_like(valid_mask)
        temp_indices = np.where(valid_mask)
        hemisphere_indices = np.where(hemisphere_mask)[0]
        temp_mask[temp_indices[0][hemisphere_indices], temp_indices[1][hemisphere_indices]] = True
        valid_mask = temp_mask
      
      if len(theta) > 0:
        # Calculate azimuth angles (vectorized)
        phi = np.arctan2(-y_cam_valid, x_cam_valid)
        
        # Apply fisheye distortion model (vectorized)
        theta_d = theta * (1 + self.k1*theta**2 + self.k2*theta**4 + 
                          self.k3*theta**6 + self.k4*theta**8)
        
        # Convert to fisheye image coordinates (vectorized)
        x_fish = self.fx * theta_d * np.cos(phi) + self.cx
        y_fish = self.fy * theta_d * np.sin(phi) + self.cy
        
        # Store coordinates in maps
        map_x[valid_mask] = x_fish
        map_y[valid_mask] = y_fish
    
    map_generation_time = time.time() - start_time
    print(f"Vectorized map generation processing time: {map_generation_time:.4f} seconds")
    
    return map_x, map_y
  
  def get_projection_maps(self, output_width: int = 1024, output_height: int = 768,
                         yaw_offset: float = 0, pitch_offset: float = 0, roll_offset: float = 0,
                         fov_horizontal: float = 90, virtual_fx: Optional[float] = None, 
                         virtual_fy: Optional[float] = None, allow_behind_camera: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get projection maps with caching.
    
    Returns cached maps if available, otherwise generates and caches new maps.
    
    Parameters:
    - output_width, output_height: dimensions of output image
    - yaw_offset, pitch_offset, roll_offset: rotation offsets in degrees
    - fov_horizontal: horizontal field of view in degrees for the virtual camera
    - virtual_fx, virtual_fy: virtual camera focal lengths (if None, calculated from FOV)
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    
    Returns:
    - map_x, map_y: projection mapping arrays
    """
    cache_key = self._generate_cache_key(output_width, output_height, yaw_offset, pitch_offset, 
                                        roll_offset, fov_horizontal, virtual_fx, virtual_fy, 
                                        allow_behind_camera)
    
    if cache_key in self._map_cache:
      print(f"Using cached projection maps: {cache_key}")
      return self._map_cache[cache_key]
    
    # Generate new maps
    map_x, map_y = self._generate_projection_maps(output_width, output_height, yaw_offset, 
                                                  pitch_offset, roll_offset, fov_horizontal, 
                                                  virtual_fx, virtual_fy, allow_behind_camera)
    
    # Cache the maps
    self._map_cache[cache_key] = (map_x, map_y)
    print(f"Cached projection maps: {cache_key}")
    
    return map_x, map_y
  
  def project(self, input_img: np.ndarray, output_width: int = 1024, output_height: int = 768, 
             yaw_offset: float = 0, pitch_offset: float = 0, roll_offset: float = 0,
             fov_horizontal: float = 90, virtual_fx: Optional[float] = None, 
             virtual_fy: Optional[float] = None, allow_behind_camera: bool = False) -> np.ndarray:
    """
    Apply perspective projection to input fisheye image.
    
    Parameters:
    - input_img: fisheye image as numpy array
    - output_width, output_height: dimensions of output image
    - yaw_offset, pitch_offset, roll_offset: rotation offsets in degrees
    - fov_horizontal: horizontal field of view in degrees for the virtual camera
    - virtual_fx, virtual_fy: virtual camera focal lengths (if None, calculated from FOV)
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    
    Returns:
    - perspective_img: projected perspective image
    """
    # Validate input image size if specified
    if self.input_image_size is not None:
      img_height, img_width = input_img.shape[:2]
      expected_width, expected_height = self.input_image_size
      if img_width != expected_width or img_height != expected_height:
        raise ValueError(f"Input image size {img_width}x{img_height} does not match expected size {expected_width}x{expected_height}")
    
    # Get projection maps (cached or generate new)
    map_x, map_y = self.get_projection_maps(output_width, output_height, yaw_offset, 
                                          pitch_offset, roll_offset, fov_horizontal, 
                                          virtual_fx, virtual_fy, allow_behind_camera)
    
    # Apply projection maps
    return apply_perspective_projection_maps(input_img, map_x, map_y)
  
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
                              yaw_offset: float, pitch_offset: float, roll_offset: float,
                              fov_horizontal: float, virtual_fx: Optional[float] = None, 
                              virtual_fy: Optional[float] = None, allow_behind_camera: bool = False):
    """Remove a specific projection from cache."""
    cache_key = self._generate_cache_key(output_width, output_height, yaw_offset, pitch_offset, 
                                        roll_offset, fov_horizontal, virtual_fx, virtual_fy, 
                                        allow_behind_camera)
    if cache_key in self._map_cache:
      del self._map_cache[cache_key]
      print(f"Removed cached projection: {cache_key}")
    else:
      print(f"Projection not in cache: {cache_key}")
