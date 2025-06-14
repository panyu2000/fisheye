"""
Fisheye Image Projection Core Modules

This package contains the core algorithms for fisheye image processing:
- Camera parameter handling and validation
- Perspective projection algorithms
- Spherical projection algorithms
"""

from .camera_params import CameraParams, parse_camera_params, parse_camera_params_dict
from .perspective_projection import PerspectiveProjection, apply_perspective_projection_maps
from .spherical_projection import SphericalProjection, apply_spherical_projection_maps

__all__ = [
    'CameraParams',
    'parse_camera_params',
    'parse_camera_params_dict',
    'PerspectiveProjection',
    'apply_perspective_projection_maps',
    'SphericalProjection',
    'apply_spherical_projection_maps'
]
