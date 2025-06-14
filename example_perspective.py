import cv2
import numpy as np
from camera_params import parse_camera_params
from perspective_projection import PerspectiveProjection

def create_custom_perspective_view():
  """
  Demonstrate creating custom perspective projections with different parameters using PerspectiveProjection class.
  """
  # Parse camera parameters
  camera_params = parse_camera_params("camera_intrinsics.yaml")
  
  # Load the fisheye image once
  fisheye_img = cv2.imread("fisheye_img.jpg")
  if fisheye_img is None:
    raise ValueError("Could not load fisheye_img.jpg")
  
  # Get image dimensions for validation
  img_height, img_width = fisheye_img.shape[:2]
  
  # Create PerspectiveProjection instance with caching capabilities and vectorized processing
  projector = PerspectiveProjection(camera_params, input_image_size=(img_width, img_height), use_vectorized=True)
  
  print("Creating custom perspective projection views using PerspectiveProjection class...")
  print("This demonstrates the flexibility and caching capabilities of the class.")
  
  # Create a wide custom view (like a security camera view)
  print("\n1. Creating wide custom view (120° horizontal)...")
  wide_custom = projector.project(
    fisheye_img,
    output_width=800, output_height=600,
    yaw_offset=30, pitch_offset=10, roll_offset=0,
    fov_horizontal=120
  )
  cv2.imwrite("fisheye_img_wide_custom.jpg", wide_custom)
  print("Saved: fisheye_img_wide_custom.jpg")
  
  # Demonstrate caching by repeating projection with same parameters
  print("\n2. Testing cache functionality - repeating same projection...")
  wide_custom_cached = projector.project(
    fisheye_img,
    output_width=800, output_height=600,
    yaw_offset=30, pitch_offset=10, roll_offset=0,
    fov_horizontal=120
  )
  
  # Verify cache worked
  if np.array_equal(wide_custom, wide_custom_cached):
    print("✓ Cache working correctly - identical results from cached projection")
  else:
    print("✗ Cache issue - results differ")
  
  # Show cache information
  cache_info = projector.get_cache_info()
  print(f"\nCache statistics:")
  print(f"  Cached projections: {cache_info['cached_projections']}")
  print(f"  Memory usage: {cache_info['memory_usage_mb']:.2f} MB")
  
  return projector, fisheye_img

def demonstrate_fov_comparison():
  """
  Create a comparison showing different field of view settings.
  """
  print("\n" + "="*60)
  print("FIELD OF VIEW COMPARISON")
  print("="*60)
  
  camera_params = parse_camera_params("camera_intrinsics.yaml")
  fisheye_img = cv2.imread("fisheye_img.jpg")
  if fisheye_img is None:
    raise ValueError("Could not load fisheye_img.jpg")
  
  img_height, img_width = fisheye_img.shape[:2]
  projector = PerspectiveProjection(camera_params, input_image_size=(img_width, img_height), use_vectorized=True)
  
  fov_angles = [20, 35, 50, 75, 90, 120, 150, 160, 170, 180]
  
  for fov in fov_angles:
    print(f"Creating {fov}° FOV comparison view...")
    fov_view = projector.project(
      fisheye_img,
      output_width=800, output_height=600,
      yaw_offset=0, pitch_offset=0, roll_offset=0,
      fov_horizontal=fov
    )
    fov_filename = f"fisheye_img_fov_{fov:03d}.jpg"
    cv2.imwrite(fov_filename, fov_view)
    print(f"  Saved: {fov_filename}")
  
  # Show final cache statistics
  cache_info = projector.get_cache_info()
  print(f"\nFOV comparison cache statistics:")
  print(f"  Cached projections: {cache_info['cached_projections']}")
  print(f"  Memory usage: {cache_info['memory_usage_mb']:.2f} MB")

def demonstrate_rotation_effects():
  """
  Create views showing different rotation effects.
  """
  print("\n" + "="*60)
  print("ROTATION EFFECTS DEMONSTRATION")
  print("="*60)
  
  camera_params = parse_camera_params("camera_intrinsics.yaml")
  fisheye_img = cv2.imread("fisheye_img.jpg")
  if fisheye_img is None:
    raise ValueError("Could not load fisheye_img.jpg")
  
  img_height, img_width = fisheye_img.shape[:2]
  projector = PerspectiveProjection(camera_params, input_image_size=(img_width, img_height), use_vectorized=True)
  
  # Yaw rotation sequence
  print("Creating yaw rotation sequence...")
  for yaw in range(0, 360, 30):
    yaw_view = projector.project(
      fisheye_img,
      output_width=600, output_height=400,
      yaw_offset=yaw, pitch_offset=0, roll_offset=0,
      fov_horizontal=70
    )
    yaw_filename = f"fisheye_img_yaw_{yaw:03d}.jpg"
    cv2.imwrite(yaw_filename, yaw_view)
  print("  Saved 12 yaw rotation views")
  
  # Pitch rotation sequence
  print("Creating pitch rotation sequence...")
  for pitch in range(-60, 61, 20):
    pitch_view = projector.project(
      fisheye_img,
      output_width=600, output_height=400,
      yaw_offset=0, pitch_offset=pitch, roll_offset=0,
      fov_horizontal=70
    )
    pitch_filename = f"fisheye_img_pitch_{pitch:+03d}.jpg"
    cv2.imwrite(pitch_filename, pitch_view)
  print("  Saved 7 pitch rotation views")
  
  # Roll rotation sequence
  print("Creating roll rotation sequence...")
  for roll in range(-45, 46, 15):
    roll_view = projector.project(
      fisheye_img,
      output_width=600, output_height=400,
      yaw_offset=0, pitch_offset=0, roll_offset=roll,
      fov_horizontal=70
    )
    roll_filename = f"fisheye_img_roll_{roll:+03d}.jpg"
    cv2.imwrite(roll_filename, roll_view)
  print("  Saved 7 roll rotation views")
  
  # Show final cache statistics
  cache_info = projector.get_cache_info()
  print(f"\nRotation effects cache statistics:")
  print(f"  Cached projections: {cache_info['cached_projections']}")
  print(f"  Memory usage: {cache_info['memory_usage_mb']:.2f} MB")

if __name__ == "__main__":
  create_custom_perspective_view()
  
  # Demonstrate FOV comparison
  demonstrate_fov_comparison()
  
  # Demonstrate rotation effects
  demonstrate_rotation_effects()
  
  print("\n" + "="*60)
  print("PERSPECTIVE PROJECTION DEMONSTRATION COMPLETE")
  print("="*60)
  print("The perspective projection method allows you to:")
  print("• Create traditional camera views with any field of view")
  print("• Simulate different lens focal lengths (wide-angle to telephoto)")
  print("• Apply 3D rotations (yaw, pitch, roll) for any viewing angle")
  print("• Generate custom aspect ratios for different applications")
  print("• Create cinematic, portrait, or social media formatted views")
  print("• Produce 360-degree scan sequences for analysis")
  print("\nKey advantages over spherical projection:")
  print("• Natural perspective distortion (like traditional cameras)")
  print("• Controllable focal length simulation")
  print("• Independent roll rotation control")
  print("• Better suited for realistic camera simulation")
  print("\nAll projections maintain geometric accuracy while providing")
  print("familiar perspective views from the fisheye source image.")
