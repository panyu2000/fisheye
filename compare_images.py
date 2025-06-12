import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_comparison():
    """
    Display original fisheye image and rectified image side by side for comparison.
    """
    # Load images
    original = cv2.imread('fisheye_img.jpg')
    rectified = cv2.imread('fisheye_img_rectified.jpg')
    
    if original is None or rectified is None:
        print("Error: Could not load one or both images")
        return
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    rectified_rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display original image
    ax1.imshow(original_rgb)
    ax1.set_title('Original Fisheye Image', fontsize=14)
    ax1.axis('off')
    
    # Display rectified image
    ax2.imshow(rectified_rgb)
    ax2.set_title('Rectified Image', fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparison saved as 'comparison.png'")
    print(f"Original image shape: {original.shape}")
    print(f"Rectified image shape: {rectified.shape}")

if __name__ == "__main__":
    display_comparison()
