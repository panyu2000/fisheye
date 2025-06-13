import cv2
import numpy as np
import os

def spherical_projection(image_path, camera_params, output_width=2048, output_height=1024, 
                        yaw_offset=0, pitch_offset=0, fov_horizontal=360, fov_vertical=180):
    """
    Project fisheye image to spherical projection (equirectangular panorama).
    
    Parameters:
    - image_path: path to fisheye image
    - camera_params: camera intrinsic parameters
    - output_width, output_height: dimensions of output panorama
    - yaw_offset, pitch_offset: rotation offsets in degrees
    - fov_horizontal, fov_vertical: field of view coverage in degrees
    """
    # Load the fisheye image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Create output image
    spherical_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Convert offsets to radians
    yaw_offset_rad = np.radians(yaw_offset)
    pitch_offset_rad = np.radians(pitch_offset)
    
    # Calculate field of view in radians
    fov_h_rad = np.radians(fov_horizontal)
    fov_v_rad = np.radians(fov_vertical)
    
    print(f"Creating spherical projection: {output_width}x{output_height}")
    print(f"FOV: {fov_horizontal}° x {fov_vertical}°")
    print(f"Offsets: yaw={yaw_offset}°, pitch={pitch_offset}°")
    
    for v in range(output_height):
        for u in range(output_width):
            # Convert output pixel to viewing direction angles
            # Map u from 0 to output_width to longitude from -fov_h/2 to +fov_h/2
            longitude = (u / output_width - 0.5) * fov_h_rad + yaw_offset_rad
            
            # Map v from 0 to output_height to latitude from +fov_v/2 to -fov_v/2  
            latitude = (0.5 - v / output_height) * fov_v_rad + pitch_offset_rad
            
            # Convert spherical angles to 3D unit vector (fisheye camera coordinate system)
            # Fisheye convention: z-axis points forward (into the scene), x-axis right, y-axis down
            x_cam = np.cos(latitude) * np.sin(longitude)  # Right direction
            y_cam = np.sin(latitude)                      # Up direction (positive for upward)
            z_cam = np.cos(latitude) * np.cos(longitude)  # Forward direction
            
            # Convert 3D vector to fisheye projection angles
            # Calculate incident angle from camera's forward axis (z-axis)
            theta = np.arccos(np.clip(z_cam, -1, 1))
            
            # # Skip if outside fisheye's hemisphere coverage
            # if theta > np.pi/2:
            #     continue
                
            # Calculate azimuth angle in image plane
            # Correct for 90-degree rotation by swapping x and y components
            phi = np.arctan2(-y_cam, x_cam)
            
            # Apply fisheye distortion model (forward projection)
            k1, k2, k3, k4 = camera_params['k1'], camera_params['k2'], camera_params['k3'], camera_params['k4']
            theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
            
            # Convert to fisheye image coordinates
            # Map from polar to Cartesian coordinates in image plane
            x_fish = camera_params['fx'] * theta_d * np.cos(phi) + camera_params['cx']
            y_fish = camera_params['fy'] * theta_d * np.sin(phi) + camera_params['cy']
            
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
                
                spherical_img[v, u] = pixel_final.astype(np.uint8)
    
    return spherical_img
