import cv2
import numpy as np
import os

def parse_camera_params(filename):
    """
    Parse camera parameters from the cameras.txt file.
    Expected format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 k3 k4
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines and empty lines
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            params = line.split()
            
            camera_id = int(params[0])
            model = params[1]
            width = int(params[2])
            height = int(params[3])
            
            # Extract intrinsic parameters
            fx = float(params[4])
            fy = float(params[5])
            cx = float(params[6])
            cy = float(params[7])
            
            # Extract distortion coefficients for fisheye model
            k1 = float(params[8])
            k2 = float(params[9])
            k3 = float(params[10])
            k4 = float(params[11])
            
            return {
                'camera_id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'k4': k4
            }
    
    raise ValueError("No valid camera parameters found in file")

def rectify_fisheye_image(image_path, camera_params, output_path=None):
    """
    Rectify a fisheye image using OpenCV fisheye model.
    """
    # Load the fisheye image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Loaded image: {img.shape}")
    
    # Set up camera matrix
    K = np.array([
        [camera_params['fx'], 0, camera_params['cx']],
        [0, camera_params['fy'], camera_params['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    
    print(f"Camera matrix K:\n{K}")
    
    # Set up distortion coefficients for fisheye model
    D = np.array([
        camera_params['k1'],
        camera_params['k2'],
        camera_params['k3'],
        camera_params['k4']
    ], dtype=np.float64)
    
    print(f"Distortion coefficients D: {D}")
    
    # Image dimensions
    img_dim = (camera_params['width'], camera_params['height'])
    
    # Estimate new camera matrix for undistorted image
    # balance=0 retains all pixels, balance=1 removes some pixels to eliminate invalid areas
    balance = 0.0
    new_size = (img_dim[0] * 2, img_dim[1] * 2)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_dim, np.eye(3), balance=balance, fov_scale=0.01
    )

    print(f"New camera matrix:\n{new_K}")

    new_K[0, 2] = img_dim[0] / 8.0  # Set new principal point x
    new_K[1, 2] = img_dim[1] / 2.0  # Set new principal point y
    
    # Create undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, img_dim, cv2.CV_16SC2
    )
    
    # Apply undistortion
    rectified_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    
    # Save rectified image
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_rectified.jpg"
    
    cv2.imwrite(output_path, rectified_img)
    print(f"Rectified image saved to: {output_path}")
    
    return rectified_img, output_path

def fisheye_to_3d_vector(x, y, camera_params):
    """
    Convert fisheye image coordinates to 3D unit vector on sphere.
    """
    # Normalize coordinates relative to principal point
    x_norm = (x - camera_params['cx']) / camera_params['fx']
    y_norm = (y - camera_params['cy']) / camera_params['fy']
    
    # Calculate radius from center
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    # Apply inverse fisheye distortion model
    theta = r  # Initial approximation
    
    # Iteratively solve for theta using Newton's method
    for _ in range(10):  # Max 10 iterations
        k1, k2, k3, k4 = camera_params['k1'], camera_params['k2'], camera_params['k3'], camera_params['k4']
        theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
        
        if abs(theta_d - r) < 1e-6:
            break
            
        # Newton's method update
        f = theta_d - r
        f_prime = (1 + 3*k1*theta**2 + 5*k2*theta**4 + 7*k3*theta**6 + 9*k4*theta**8)
        theta = theta - f / f_prime if f_prime != 0 else theta
    
    # Convert to 3D coordinates
    if r > 1e-6:
        phi = np.arctan2(y_norm, x_norm)  # Azimuth angle
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # 3D unit vector on sphere
        x_3d = sin_theta * np.cos(phi)
        y_3d = sin_theta * np.sin(phi)
        z_3d = cos_theta
    else:
        # Handle center point
        x_3d, y_3d, z_3d = 0, 0, 1
    
    return np.array([x_3d, y_3d, z_3d])

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
            
            # Skip if outside fisheye's hemisphere coverage
            if theta > np.pi/2:
                continue
                
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

def perspective_projection(image_path, camera_params, output_width=1024, output_height=768,
                          yaw_offset=0, pitch_offset=0, roll_offset=0, 
                          fov_horizontal=90, virtual_fx=None, virtual_fy=None):
    """
    Project fisheye image to perspective projection (like a traditional camera view).
    
    Parameters:
    - image_path: path to fisheye image
    - camera_params: camera intrinsic parameters
    - output_width, output_height: dimensions of output image
    - yaw_offset, pitch_offset, roll_offset: rotation offsets in degrees
    - fov_horizontal: horizontal field of view in degrees for the virtual camera
    - virtual_fx, virtual_fy: virtual camera focal lengths (if None, calculated from FOV)
    """
    # Load the fisheye image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Create output image
    perspective_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
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
    
    print(f"Creating perspective projection: {output_width}x{output_height}")
    print(f"Virtual camera FOV: {fov_horizontal}°")
    print(f"Virtual camera params: fx={virtual_fx:.1f}, fy={virtual_fy:.1f}")
    print(f"Rotation: yaw={yaw_offset}°, pitch={pitch_offset}°, roll={roll_offset}°")
    
    # Create rotation matrices using standard camera coordinate conventions
    # Yaw: rotation around Y-axis (left/right turn)
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    # Pitch: rotation around X-axis (up/down tilt)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
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
            
            # Skip if ray points backward (negative z)
            if z_cam <= 0:
                continue
            
            # Convert 3D direction to fisheye projection angles
            theta = np.arccos(np.clip(z_cam, -1, 1))
            
            # Skip if outside fisheye's hemisphere coverage
            if theta > np.pi/2:
                continue
            
            # Calculate azimuth angle
            phi = np.arctan2(-y_cam, x_cam)
            
            # Apply fisheye distortion model (forward projection)
            k1, k2, k3, k4 = camera_params['k1'], camera_params['k2'], camera_params['k3'], camera_params['k4']
            theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
            
            # Convert to fisheye image coordinates
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
                
                perspective_img[v, u] = pixel_final.astype(np.uint8)
    
    return perspective_img

def create_perspective_projections(image_path, camera_params):
    """
    Create multiple perspective projections with different viewing angles and FOVs.
    """
    base_name = os.path.splitext(image_path)[0]
    outputs = []
    
    # Standard front view (like a normal camera)
    print("\nCreating standard perspective view...")
    standard_view = perspective_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=0, roll_offset=0,
        fov_horizontal=90
    )
    standard_path = f"{base_name}_perspective_standard.jpg"
    cv2.imwrite(standard_path, standard_view)
    print(f"Standard perspective saved to: {standard_path}")
    outputs.append(standard_path)
    
    # Wide angle view
    print("Creating wide-angle perspective view...")
    wide_view = perspective_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=0, roll_offset=0,
        fov_horizontal=120
    )
    wide_path = f"{base_name}_perspective_wide.jpg"
    cv2.imwrite(wide_path, wide_view)
    print(f"Wide-angle perspective saved to: {wide_path}")
    outputs.append(wide_path)
    
    # Narrow telephoto view
    print("Creating telephoto perspective view...")
    telephoto_view = perspective_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=0, roll_offset=0,
        fov_horizontal=45
    )
    telephoto_path = f"{base_name}_perspective_telephoto.jpg"
    cv2.imwrite(telephoto_path, telephoto_view)
    print(f"Telephoto perspective saved to: {telephoto_path}")
    outputs.append(telephoto_path)
    
    # Side view (90 degrees rotated)
    print("Creating side perspective view...")
    side_view = perspective_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=90, pitch_offset=0, roll_offset=0,
        fov_horizontal=90
    )
    side_path = f"{base_name}_perspective_side.jpg"
    cv2.imwrite(side_path, side_view)
    print(f"Side perspective saved to: {side_path}")
    outputs.append(side_path)
    
    # Looking up
    print("Creating upward perspective view...")
    upward_view = perspective_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=30, roll_offset=0,
        fov_horizontal=90
    )
    upward_path = f"{base_name}_perspective_upward.jpg"
    cv2.imwrite(upward_path, upward_view)
    print(f"Upward perspective saved to: {upward_path}")
    outputs.append(upward_path)
    
    # Looking down
    print("Creating downward perspective view...")
    downward_view = perspective_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=-30, roll_offset=0,
        fov_horizontal=90
    )
    downward_path = f"{base_name}_perspective_downward.jpg"
    cv2.imwrite(downward_path, downward_view)
    print(f"Downward perspective saved to: {downward_path}")
    outputs.append(downward_path)
    
    # Tilted view (with roll)
    print("Creating tilted perspective view...")
    tilted_view = perspective_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=0, roll_offset=15,
        fov_horizontal=90
    )
    tilted_path = f"{base_name}_perspective_tilted.jpg"
    cv2.imwrite(tilted_path, tilted_view)
    print(f"Tilted perspective saved to: {tilted_path}")
    outputs.append(tilted_path)
    
    return outputs

def create_spherical_projections(image_path, camera_params):
    """
    Create multiple spherical projections with different viewing angles.
    """
    base_name = os.path.splitext(image_path)[0]
    
    # Full equirectangular panorama
    print("\nCreating full equirectangular panorama...")
    full_panorama = spherical_projection(
        image_path, camera_params, 
        output_width=2048, output_height=1024,
        fov_horizontal=360, fov_vertical=180
    )
    panorama_path = f"{base_name}_spherical_panorama.jpg"
    cv2.imwrite(panorama_path, full_panorama)
    print(f"Full panorama saved to: {panorama_path}")
    
    # Front view (wide angle)
    print("Creating front view projection...")
    front_view = spherical_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=0, pitch_offset=0,
        fov_horizontal=120, fov_vertical=90
    )
    front_path = f"{base_name}_spherical_front.jpg"
    cv2.imwrite(front_path, front_view)
    print(f"Front view saved to: {front_path}")
    
    # Side view (90 degrees rotated)
    print("Creating side view projection...")
    side_view = spherical_projection(
        image_path, camera_params,
        output_width=1024, output_height=768,
        yaw_offset=90, pitch_offset=0,
        fov_horizontal=120, fov_vertical=90
    )
    side_path = f"{base_name}_spherical_side.jpg"
    cv2.imwrite(side_path, side_view)
    print(f"Side view saved to: {side_path}")
    
    # Top-down view
    print("Creating top-down view projection...")
    top_view = spherical_projection(
        image_path, camera_params,
        output_width=1024, output_height=1024,
        yaw_offset=0, pitch_offset=45,
        fov_horizontal=180, fov_vertical=180
    )
    top_path = f"{base_name}_spherical_top.jpg"
    cv2.imwrite(top_path, top_view)
    print(f"Top view saved to: {top_path}")
    
    return [panorama_path, front_path, side_path, top_path]

def main():
    # File paths
    camera_file = "cameras.txt"
    input_image = "fisheye_img.jpg"
    
    try:
        # Parse camera parameters
        print("Parsing camera parameters...")
        camera_params = parse_camera_params(camera_file)
        print(f"Camera parameters loaded:")
        print(f"  Model: {camera_params['model']}")
        print(f"  Image size: {camera_params['width']}x{camera_params['height']}")
        print(f"  Focal length: fx={camera_params['fx']}, fy={camera_params['fy']}")
        print(f"  Principal point: cx={camera_params['cx']}, cy={camera_params['cy']}")
        print(f"  Distortion: k1={camera_params['k1']}, k2={camera_params['k2']}, k3={camera_params['k3']}, k4={camera_params['k4']}")
        
        # Rectify the fisheye image using OpenCV method
        print("\nRectifying fisheye image using OpenCV...")
        rectified_img, output_path = rectify_fisheye_image(input_image, camera_params)
        
        print(f"\nOpenCV rectification completed successfully!")
        print(f"Original image: {input_image}")
        print(f"Rectified image: {output_path}")
        
        # Create spherical projections
        print("\n" + "="*60)
        print("CREATING SPHERICAL PROJECTIONS")
        print("="*60)
        spherical_outputs = create_spherical_projections(input_image, camera_params)
        
        print(f"\nSpherical projection completed successfully!")
        print("Generated spherical files:")
        for i, output_file in enumerate(spherical_outputs, 1):
            print(f"  {i}. {output_file}")
        
        # Create perspective projections
        print("\n" + "="*60)
        print("CREATING PERSPECTIVE PROJECTIONS")
        print("="*60)
        perspective_outputs = create_perspective_projections(input_image, camera_params)
        
        print(f"\nPerspective projection completed successfully!")
        print("Generated perspective files:")
        for i, output_file in enumerate(perspective_outputs, 1):
            print(f"  {i}. {output_file}")
        
        print(f"\nAll projections completed successfully!")
        print(f"Total files generated: {len(spherical_outputs) + len(perspective_outputs)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
