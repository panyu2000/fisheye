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
import yaml


class CameraParams:
  """
  Camera parameters class for fisheye camera model.
  
  This class holds all camera intrinsic parameters and distortion coefficients
  for the OpenCV fisheye camera model, following the format defined in cameras.txt.
  """
  
  def __init__(self, camera_id=None, model=None, width=None, height=None,
               fx=None, fy=None, cx=None, cy=None,
               k1=None, k2=None, k3=None, k4=None):
    """
    Initialize camera parameters.
    
    Parameters:
    - camera_id: unique identifier for the camera
    - model: camera model type (e.g., 'OPENCV_FISHEYE')
    - width: image width in pixels
    - height: image height in pixels
    - fx, fy: focal lengths in pixels
    - cx, cy: principal point coordinates in pixels
    - k1, k2, k3, k4: fisheye distortion coefficients
    """
    self.camera_id = camera_id
    self.model = model
    self.width = width
    self.height = height
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.k4 = k4
  
  def to_dict(self):
    """
    Convert camera parameters to dictionary format.
    
    Returns:
    Dictionary containing all camera parameters, compatible with existing code.
    """
    return {
      'camera_id': self.camera_id,
      'model': self.model,
      'width': self.width,
      'height': self.height,
      'fx': self.fx,
      'fy': self.fy,
      'cx': self.cx,
      'cy': self.cy,
      'k1': self.k1,
      'k2': self.k2,
      'k3': self.k3,
      'k4': self.k4
    }
  
  def get_camera_matrix(self):
    """
    Get OpenCV camera matrix K.
    
    Returns:
    3x3 numpy array representing the camera intrinsic matrix.
    """
    return np.array([
      [self.fx, 0, self.cx],
      [0, self.fy, self.cy],
      [0, 0, 1]
    ], dtype=np.float64)
  
  def get_distortion_coefficients(self):
    """
    Get fisheye distortion coefficients.
    
    Returns:
    4-element numpy array with fisheye distortion coefficients [k1, k2, k3, k4].
    """
    return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)
  
  def get_image_size(self):
    """
    Get image dimensions as tuple.
    
    Returns:
    Tuple (width, height) of image dimensions.
    """
    return (self.width, self.height)
  
  def validate(self):
    """
    Validate camera parameters for reasonable ranges.
    
    Raises:
    ValueError if any parameter is invalid or out of reasonable range.
    """
    if self.width <= 0 or self.height <= 0:
      raise ValueError(f"Invalid image dimensions: {self.width}x{self.height}")
    
    if self.fx <= 0 or self.fy <= 0:
      raise ValueError(f"Invalid focal lengths: fx={self.fx}, fy={self.fy}")
    
    if not (0 <= self.cx <= self.width) or not (0 <= self.cy <= self.height):
      raise ValueError(f"Principal point outside image bounds: cx={self.cx}, cy={self.cy}")
    
    # Check if distortion coefficients are in reasonable ranges
    # Typical fisheye distortion coefficients are usually small values
    distortion_values = [self.k1, self.k2, self.k3, self.k4]
    if any(abs(k) > 10.0 for k in distortion_values):
      raise ValueError(f"Distortion coefficients seem unreasonable: {distortion_values}")
  
  def __str__(self):
    """String representation of camera parameters."""
    return (f"CameraParams(id={self.camera_id}, model={self.model}, "
            f"size={self.width}x{self.height}, "
            f"fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
            f"k1={self.k1:.6f}, k2={self.k2:.6f}, k3={self.k3:.6f}, k4={self.k4:.6f})")
  
  def __repr__(self):
    """Detailed representation of camera parameters."""
    return self.__str__()


def parse_camera_params(filename):
  """
  Parse camera parameters from YAML file and return CameraParams object.
  
  Expected YAML format with OpenCV intrinsics structure.
  
  Parameters:
  - filename: path to YAML camera parameters file
  
  Returns:
  CameraParams object with loaded parameters.
  
  Raises:
  ValueError if file format is invalid or parameters are missing.
  FileNotFoundError if camera file doesn't exist.
  """
  try:
    with open(filename, 'r') as f:
      data = yaml.safe_load(f)
  except FileNotFoundError:
    raise FileNotFoundError(f"Camera parameters file not found: {filename}")
  except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML format in file '{filename}': {e}")
  
  try:
    # Extract basic parameters
    width = data['image_width']
    height = data['image_height']
    camera_name = data.get('camera_name', 'unknown')
    model = data.get('distortion_model', 'fisheye').upper()
    
    # Extract camera matrix data
    camera_matrix_data = data['camera_matrix']['data']
    if len(camera_matrix_data) != 9:
      raise ValueError("Camera matrix must have 9 elements")
    
    # Camera matrix is stored row-wise: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    fx = camera_matrix_data[0]
    cx = camera_matrix_data[2]
    fy = camera_matrix_data[4]
    cy = camera_matrix_data[5]
    
    # Extract distortion coefficients
    distortion_data = data['distortion_coefficients']['data']
    if len(distortion_data) != 4:
      raise ValueError("Fisheye distortion coefficients must have 4 elements")
    
    k1, k2, k3, k4 = distortion_data
    
    # Create CameraParams object
    camera_params = CameraParams(
      camera_id=camera_name,
      model=model,
      width=width,
      height=height,
      fx=fx,
      fy=fy,
      cx=cx,
      cy=cy,
      k1=k1,
      k2=k2,
      k3=k3,
      k4=k4
    )
    
    # Validate parameters
    camera_params.validate()
    
    return camera_params
    
  except KeyError as e:
    raise ValueError(f"Missing required parameter in YAML file: {e}")
  except (TypeError, ValueError) as e:
    raise ValueError(f"Invalid parameter format in YAML file: {e}")


def parse_camera_params_dict(filename):
  """
  Parse camera parameters from file and return dictionary format.
  
  This function provides backward compatibility with existing code
  that expects camera parameters as a dictionary.
  
  Parameters:
  - filename: path to camera parameters file
  
  Returns:
  Dictionary containing camera parameters.
  """
  camera_params = parse_camera_params(filename)
  return camera_params.to_dict()


if __name__ == "__main__":
  # Example usage and testing
  try:
    print("Testing camera parameter parsing...")
    
    # Parse camera parameters from YAML file
    camera_params = parse_camera_params("camera_intrinsics.yaml")
    print(f"Loaded camera parameters: {camera_params}")
    
    # Test dictionary conversion
    params_dict = camera_params.to_dict()
    print(f"Dictionary format: {params_dict}")
    
    # Test OpenCV matrices
    K = camera_params.get_camera_matrix()
    D = camera_params.get_distortion_coefficients()
    print(f"Camera matrix K:\n{K}")
    print(f"Distortion coefficients D: {D}")
    
    # Test image size
    img_size = camera_params.get_image_size()
    print(f"Image size: {img_size}")
    
    print("Camera parameter parsing test completed successfully!")
    
  except Exception as e:
    print(f"Error during testing: {e}")
