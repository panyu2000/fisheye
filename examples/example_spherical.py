import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.camera_params import parse_camera_params
from src.spherical_projection import SphericalProjection

def create_custom_spherical_view():
  """
  Demonstrate creating custom spherical projections with different parameters using SphericalProjection class.
  """
  # Parse camera parameters
  camera_params = parse_camera_params("config/camera_intrinsics.yaml")
  
  # Load the fisheye image once
  fisheye_img = cv2.imread("data/fisheye_img.jpg")
  if fisheye_img is None:
    raise ValueError("Could not load data/fisheye_img.jpg")
  
  # Get image dimensions for validation
  img_height, img_width = fisheye_img.shape[:2]
  
  # Create SphericalProjection instance with caching capabilities and vectorized processing
  projector = SphericalProjection(camera_params, input_image_size=(img_width, img_height), use_vectorized=True)
  
  print("Creating custom spherical projection views using SphericalProjection class...")
  print("This demonstrates the flexibility and caching capabilities of the class.")
  
  # Create a wide panoramic view (like a security camera view)
  print("\n1. Creating wide panoramic view (220° horizontal)...")
  wide_panorama = projector.project(
    fisheye_img,
    output_width=1200, output_height=500,
    yaw_offset=0, pitch_offset=0,
    fov_horizontal=220, fov_vertical=120
  )
  cv2.imwrite("output/spherical/fisheye_img_wide_panorama.jpg", wide_panorama)
  print("Saved: output/spherical/fisheye_img_wide_panorama.jpg")
  
  # Create a narrow field of view (like a normal camera)
  print("\n2. Creating narrow field of view (60° perspective)...")
  narrow_view = projector.project(
    fisheye_img,
    output_width=800, output_height=600,
    yaw_offset=0, pitch_offset=0,
    fov_horizontal=60, fov_vertical=45
  )
  cv2.imwrite("output/spherical/fisheye_img_narrow_view.jpg", narrow_view)
  print("Saved: output/spherical/fisheye_img_narrow_view.jpg")
  
  # Create a tilted view (looking up)
  print("\n3. Creating tilted upward view...")
  upward_view = projector.project(
    fisheye_img,
    output_width=1024, output_height=768,
    yaw_offset=0, pitch_offset=30,
    fov_horizontal=120, fov_vertical=90
  )
  cv2.imwrite("output/spherical/fisheye_img_upward_view.jpg", upward_view)
  print("Saved: output/spherical/fisheye_img_upward_view.jpg")
  
  # Create a rotated side view
  print("\n4. Creating rotated view (looking backward)...")
  backward_view = projector.project(
    fisheye_img,
    output_width=1024, output_height=768,
    yaw_offset=180, pitch_offset=0,
    fov_horizontal=120, fov_vertical=90
  )
  cv2.imwrite("output/spherical/fisheye_img_backward_view.jpg", backward_view)
  print("Saved: output/spherical/fisheye_img_backward_view.jpg")
  
  # Create a full 360° spherical panorama (allow behind camera)
  print("\n5. Creating full 360° spherical panorama...")
  full_spherical = projector.project(
    fisheye_img,
    output_width=2048, output_height=1024,
    yaw_offset=0, pitch_offset=0,
    fov_horizontal=360, fov_vertical=180,
    allow_behind_camera=True
  )
  cv2.imwrite("output/spherical/fisheye_img_full_spherical.jpg", full_spherical)
  print("Saved: output/spherical/fisheye_img_full_spherical.jpg")
  
  # Compare hemisphere vs full spherical projection
  print("\n6. Comparing hemisphere vs full spherical projection...")
  hemisphere_view = projector.project(
    fisheye_img,
    output_width=1024, output_height=512,
    yaw_offset=0, pitch_offset=0,
    fov_horizontal=180, fov_vertical=90,
    allow_behind_camera=False  # Traditional hemisphere only
  )
  cv2.imwrite("output/spherical/fisheye_img_hemisphere.jpg", hemisphere_view)
  print("Saved: output/spherical/fisheye_img_hemisphere.jpg (hemisphere only)")
  
  full_180_view = projector.project(
    fisheye_img,
    output_width=1024, output_height=512,
    yaw_offset=0, pitch_offset=0,
    fov_horizontal=180, fov_vertical=90,
    allow_behind_camera=True  # Include behind camera content
  )
  cv2.imwrite("output/spherical/fisheye_img_full_180.jpg", full_180_view)
  print("Saved: output/spherical/fisheye_img_full_180.jpg (with behind camera content)")
  
  # Demonstrate caching by repeating a projection with same parameters
  print("\n7. Testing cache functionality - repeating first projection...")
  wide_panorama_cached = projector.project(
    fisheye_img,
    output_width=1200, output_height=500,
    yaw_offset=0, pitch_offset=0,
    fov_horizontal=220, fov_vertical=120
  )
  
  # Verify cache worked (results should be identical)
  if np.array_equal(wide_panorama, wide_panorama_cached):
    print("✓ Cache working correctly - identical results from cached projection")
  else:
    print("✗ Cache issue - results differ")
  
  # Show cache information
  cache_info = projector.get_cache_info()
  print(f"\nCache statistics:")
  print(f"  Cached projections: {cache_info['cached_projections']}")
  print(f"  Memory usage: {cache_info['memory_usage_mb']:.2f} MB")
  
  print("\nCustom spherical projections completed!")
  print("Total files created: 7")
  
  return [
    "fisheye_img_wide_panorama.jpg",
    "fisheye_img_narrow_view.jpg", 
    "fisheye_img_upward_view.jpg",
    "fisheye_img_backward_view.jpg",
    "fisheye_img_full_spherical.jpg",
    "fisheye_img_hemisphere.jpg",
    "fisheye_img_full_180.jpg"
  ]

if __name__ == "__main__":
  custom_files = create_custom_spherical_view()
  
  print("\n" + "="*60)
  print("SPHERICAL PROJECTION DEMONSTRATION COMPLETE")
  print("="*60)
  print("The spherical projection method allows you to:")
  print("• Create panoramic views with any field of view")
  print("• Look in any direction (yaw/pitch offsets)")
  print("• Generate different perspective views from the same fisheye image")
  print("• Create virtual camera views with custom parameters")
  print("• Control hemisphere vs full spherical coverage (allow_behind_camera)")
  print("• Generate traditional fisheye projections or full 360° spherical views")
  print("\nKey demonstrations:")
  print("• Wide panoramic views for security/surveillance applications")
  print("• Narrow field views for traditional camera simulation")
  print("• Full 360° spherical panoramas for VR/immersive content")
  print("• Hemisphere vs full spherical comparison showing coverage differences")
  print("\nAll projection methods preserve the geometric relationships")
  print("from the original fisheye image while providing different viewing perspectives.")
