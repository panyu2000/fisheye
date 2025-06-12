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
        
        # Rectify the fisheye image
        print("\nRectifying fisheye image...")
        rectified_img, output_path = rectify_fisheye_image(input_image, camera_params)
        
        print(f"\nRectification completed successfully!")
        print(f"Original image: {input_image}")
        print(f"Rectified image: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
