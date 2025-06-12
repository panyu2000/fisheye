import cv2
import numpy as np
from fisheye_rectify import parse_camera_params, spherical_projection

def create_custom_spherical_view():
    """
    Demonstrate creating custom spherical projections with different parameters.
    """
    # Parse camera parameters
    camera_params = parse_camera_params("cameras.txt")
    
    print("Creating custom spherical projection views...")
    print("This demonstrates the flexibility of the spherical projection method.")
    
    # Create a wide panoramic view (like a security camera view)
    print("\n1. Creating wide panoramic view (270° horizontal)...")
    wide_panorama = spherical_projection(
        "fisheye_img.jpg", camera_params,
        output_width=1200, output_height=800,
        yaw_offset=0, pitch_offset=-30,
        fov_horizontal=180, fov_vertical=120
    )
    cv2.imwrite("fisheye_img_wide_panorama.jpg", wide_panorama)
    print("Saved: fisheye_img_wide_panorama.jpg")
    
    # Create a narrow field of view (like a normal camera)
    print("\n2. Creating narrow field of view (60° perspective)...")
    narrow_view = spherical_projection(
        "fisheye_img.jpg", camera_params,
        output_width=800, output_height=600,
        yaw_offset=0, pitch_offset=0,
        fov_horizontal=60, fov_vertical=45
    )
    cv2.imwrite("fisheye_img_narrow_view.jpg", narrow_view)
    print("Saved: fisheye_img_narrow_view.jpg")
    
    # Create a tilted view (looking up)
    print("\n3. Creating tilted upward view...")
    upward_view = spherical_projection(
        "fisheye_img.jpg", camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=30,
        fov_horizontal=120, fov_vertical=90
    )
    cv2.imwrite("fisheye_img_upward_view.jpg", upward_view)
    print("Saved: fisheye_img_upward_view.jpg")
    
    # Create a rotated side view
    print("\n4. Creating rotated view (looking backward)...")
    backward_view = spherical_projection(
        "fisheye_img.jpg", camera_params,
        output_width=1024, output_height=768,
        yaw_offset=180, pitch_offset=0,
        fov_horizontal=120, fov_vertical=90
    )
    cv2.imwrite("fisheye_img_backward_view.jpg", backward_view)
    print("Saved: fisheye_img_backward_view.jpg")
    
    print("\nCustom spherical projections completed!")
    print("Total files created: 4")
    
    return [
        "fisheye_img_wide_panorama.jpg",
        "fisheye_img_narrow_view.jpg", 
        "fisheye_img_upward_view.jpg",
        "fisheye_img_backward_view.jpg"
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
    print("\nAll projection methods preserve the geometric relationships")
    print("from the original fisheye image while providing different viewing perspectives.")
