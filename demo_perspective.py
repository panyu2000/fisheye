import cv2
import numpy as np
from fisheye_rectify import parse_camera_params, perspective_projection

def create_custom_perspective_view():
    """
    Demonstrate creating custom perspective projections with different parameters.
    """
    # Parse camera parameters
    camera_params = parse_camera_params("cameras.txt")
    
    print("Creating custom perspective projection views...")
    print("This demonstrates the flexibility of the perspective projection method.")
    
    # Create a wide panoramic view (like a security camera view)
    print("\n1. Creating wide custom view (120° horizontal)...")
    wide_custom = perspective_projection(
        "fisheye_img.jpg", camera_params,
        output_width=800, output_height=600,
        yaw_offset=30, pitch_offset=10, roll_offset=0,
        fov_horizontal=120
    )
    cv2.imwrite("fisheye_img_wide_custom.jpg", wide_custom)
    print("Saved: fisheye_img_wide_cstom.jpg")

def demonstrate_fov_comparison():
    """
    Create a comparison showing different field of view settings.
    """
    print("\n" + "="*60)
    print("FIELD OF VIEW COMPARISON")
    print("="*60)
    
    camera_params = parse_camera_params("cameras.txt")
    fov_angles = [20, 35, 50, 75, 90, 120, 150, 160, 170, 180]
    
    for fov in fov_angles:
        print(f"Creating {fov}° FOV comparison view...")
        fov_view = perspective_projection(
            "fisheye_img.jpg", camera_params,
            output_width=800, output_height=600,
            yaw_offset=0, pitch_offset=0, roll_offset=0,
            fov_horizontal=fov
        )
        fov_filename = f"fisheye_img_fov_{fov:03d}.jpg"
        cv2.imwrite(fov_filename, fov_view)
        print(f"  Saved: {fov_filename}")

def demonstrate_rotation_effects():
    """
    Create views showing different rotation effects.
    """
    print("\n" + "="*60)
    print("ROTATION EFFECTS DEMONSTRATION")
    print("="*60)
    
    camera_params = parse_camera_params("cameras.txt")
    
    # Yaw rotation sequence
    print("Creating yaw rotation sequence...")
    for yaw in range(0, 360, 30):
        yaw_view = perspective_projection(
            "fisheye_img.jpg", camera_params,
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
        pitch_view = perspective_projection(
            "fisheye_img.jpg", camera_params,
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
        roll_view = perspective_projection(
            "fisheye_img.jpg", camera_params,
            output_width=600, output_height=400,
            yaw_offset=0, pitch_offset=0, roll_offset=roll,
            fov_horizontal=70
        )
        roll_filename = f"fisheye_img_roll_{roll:+03d}.jpg"
        cv2.imwrite(roll_filename, roll_view)
    print("  Saved 7 roll rotation views")

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
