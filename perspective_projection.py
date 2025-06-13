import cv2
import numpy as np
import os
from typing import Tuple, Dict, Optional
from camera_params import CameraParams

def apply_perspective_projection_maps(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
  """
  Apply perspective projection mapping to fisheye image using pre-generated maps.
  
  Parameters:
  - img: fisheye image as numpy array
  - map_x: array of x coordinates in fisheye image for each output pixel
  - map_y: array of y coordinates in fisheye image for each output pixel
  
  Returns:
  - perspective_img: projected perspective image
  """
  if img is None:
    raise ValueError("Input image is None")
  
  height, width = img.shape[:2]
  output_height, output_width = map_x.shape
  
  # Create output image
  perspective_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
  
  print(f"Applying perspective projection maps to create {output_width}x{output_height} image")
  
  for v in range(output_height):
    for u in range(output_width):
      x_fish = map_x[v, u]
      y_fish = map_y[v, u]
      
      # Check if coordinates are within image bounds
      if 0 <= x_fish < width and 0 <= y_fish < height:
        # Bilinear interpolation
        x_floor, y_floor = int(x_fish), int(y_fish)
        x_ceil, y_ceil = min(x_floor + 1, width - 1), min(y_floor + 1, height - 1)
        
        # Interpolation weights
        wx = x_fish - x_floor
        wy = y_fish - y_floor
        
        # Sample pixel values
        pixel_tl = img[y_floor, x_floor]
        pixel_tr = img[y_floor, x_ceil]
        pixel_bl = img[y_ceil, x_floor]
        pixel_br = img[y_ceil, x_ceil]
        
        # Bilinear interpolation
        pixel_top = (1 - wx) * pixel_tl + wx * pixel_tr
        pixel_bottom = (1 - wx) * pixel_bl + wx * pixel_br
        pixel_final = (1 - wy) * pixel_top + wy * pixel_bottom
        
        perspective_img[v, u] = pixel_final.astype(np.uint8)
  
  return perspective_img

class PerspectiveProjection:
  """
  Perspective projection processor for fisheye cameras with caching capabilities.
  
  This class represents a fisheye camera and provides efficient perspective projection
  functionality with automatic caching of projection maps for improved performance
  when processing multiple images with the same projection parameters.
  """
  
  def __init__(self, camera_params: CameraParams, input_image_size: Optional[Tuple[int, int]] = None):
    """
    Initialize PerspectiveProjection with camera parameters.
    
    Parameters:
    - camera_params: CameraParams object
    - input_image_size: (width, height) of input fisheye images, optional for validation
    """
    self.camera_params = camera_params
    self.input_image_size = input_image_size
    
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
    
    This is an internal method that uses the stored camera parameters.
    
    Parameters:
    - allow_behind_camera: if True, include rays pointing backward (z < 0) in the projection maps
    """
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
    
    # Combined rotation matrix (apply in order: yaw, pitch, roll)
    R_combined = R_roll @ R_pitch @ R_yaw
    
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
