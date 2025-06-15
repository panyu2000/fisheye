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

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
from typing import Dict, Union, Optional, List
import datetime
import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.camera_params import parse_camera_params
from src.perspective_projection import PerspectiveProjection
from src.spherical_projection import SphericalProjection
from src.cache_manager import CacheManager

class FisheyeProjectionGUI:
  def __init__(self, root: tk.Tk) -> None:
    self.root = root
    self.root.title("Fisheye Projection Tool - Perspective & Spherical")
    self.root.geometry("1600x1000")
    
    # Load camera parameters and scan for images
    try:
      self.camera_params = parse_camera_params("config/camera_intrinsics.yaml")
      
      # Scan for images matching camera name prefix
      self.available_images = self.scan_data_folder()
      
      # Set default image (try fisheye_img.jpg first, then first available)
      self.current_image_path = "data/fisheye_img.jpg"
      if not os.path.exists(self.current_image_path) and self.available_images:
        self.current_image_path = self.available_images[0]
      
      # Load current image
      self.fisheye_img = cv2.imread(self.current_image_path)
      if self.fisheye_img is None:
        if self.available_images:
          raise FileNotFoundError(f"Could not load any images from data folder. Found files: {self.available_images}")
        else:
          raise FileNotFoundError("No matching images found in data folder")
        
      # Get image dimensions for validation
      img_height, img_width = self.fisheye_img.shape[:2]
      
      # Create shared cache manager with 256MB limit and LRU eviction
      self.shared_cache = CacheManager(max_memory_mb=256.0)
      
      # Create projection instances with shared caching capabilities
      self.perspective_projector = PerspectiveProjection(
        self.camera_params, 
        use_vectorized=True,
        cache_manager=self.shared_cache
      )
      self.spherical_projector = SphericalProjection(
        self.camera_params, 
        use_vectorized=True,
        cache_manager=self.shared_cache
      )
      
    except Exception as e:
      messagebox.showerror("Error", f"Failed to load files: {e}")
      return
    
    # Initialize parameters for both projection types
    self.init_parameters()
    
    # Threading queues for both projections
    self.init_threading()
    
    # Setup UI
    self.setup_ui()
    
    # Setup keyboard controls
    self.setup_keyboard_controls()
    
    # Start initial processing
    self.update_images()
    
    # Start periodic check for results
    self.root.after(100, self.check_results)
  
  def init_parameters(self) -> None:
    """Initialize parameters for both projection types."""
    # Perspective projection parameters
    self.perspective_params = {
      'output_width': tk.IntVar(value=800),
      'output_height': tk.IntVar(value=800),
      'yaw_offset': tk.DoubleVar(value=0.0),
      'pitch_offset': tk.DoubleVar(value=0.0),
      'roll_offset': tk.DoubleVar(value=0.0),
      'fov_horizontal': tk.DoubleVar(value=90.0),
      'virtual_fx': tk.DoubleVar(value=0.0),
      'virtual_fy': tk.DoubleVar(value=0.0),
      'allow_behind_camera': tk.BooleanVar(value=True)
    }
    
    # Spherical projection parameters
    self.spherical_params = {
      'output_width': tk.IntVar(value=2048),
      'output_height': tk.IntVar(value=1024),
      'yaw_offset': tk.DoubleVar(value=0.0),
      'pitch_offset': tk.DoubleVar(value=0.0),
      'fov_horizontal': tk.DoubleVar(value=360.0),
      'fov_vertical': tk.DoubleVar(value=180.0),
      'allow_behind_camera': tk.BooleanVar(value=True)
    }
  
  def scan_data_folder(self) -> List[str]:
    """Scan data folder for images matching camera name prefix."""
    camera_name = self.camera_params.camera_id or "fisheye_camera"
    data_folder = "data"
    
    # Supported image extensions
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    
    # Find all image files
    all_images = []
    for ext in extensions:
      all_images.extend(glob.glob(os.path.join(data_folder, ext)))
      all_images.extend(glob.glob(os.path.join(data_folder, ext.upper())))
    
    # Filter images that start with camera name
    matching_images = []
    for img_path in all_images:
      filename = os.path.basename(img_path)
      # Check if filename starts with camera name (case insensitive)
      if filename.lower().startswith(camera_name.lower()):
        matching_images.append(img_path)
    
    # Sort images for consistent ordering
    matching_images.sort()
    
    # Also include default fisheye_img.jpg if it exists
    default_path = os.path.join(data_folder, "fisheye_img.jpg")
    if os.path.exists(default_path) and default_path not in matching_images:
      matching_images.insert(0, default_path)
    
    return matching_images

  def init_threading(self) -> None:
    """Initialize threading components for both projections."""
    self.perspective_processing_queue = queue.Queue()
    self.perspective_result_queue = queue.Queue()
    self.perspective_processing_thread = None
    self.perspective_pending_update_id = None
    
    self.spherical_processing_queue = queue.Queue()
    self.spherical_result_queue = queue.Queue()
    self.spherical_processing_thread = None
    self.spherical_pending_update_id = None
  
  def setup_ui(self) -> None:
    """Setup the main tabbed interface."""
    # Main container
    main_frame = ttk.Frame(self.root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    self.root.columnconfigure(0, weight=1)
    self.root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    
    # Create notebook for tabs
    self.notebook = ttk.Notebook(main_frame)
    self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Create perspective projection tab
    self.perspective_frame = ttk.Frame(self.notebook)
    self.notebook.add(self.perspective_frame, text="Perspective Projection")
    
    # Create spherical projection tab
    self.spherical_frame = ttk.Frame(self.notebook)
    self.notebook.add(self.spherical_frame, text="Spherical Projection")
    
    # Setup individual tabs
    self.setup_perspective_tab()
    self.setup_spherical_tab()
    
    # Bind tab change event
    self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
  
  def setup_perspective_tab(self) -> None:
    """Setup the perspective projection tab."""
    # Configure grid weights
    self.perspective_frame.columnconfigure(0, weight=0)  # Controls fixed width
    self.perspective_frame.columnconfigure(1, weight=1)  # Images take remaining space
    self.perspective_frame.rowconfigure(0, weight=1)
    
    # Left panel for controls
    control_frame = ttk.LabelFrame(self.perspective_frame, text="Perspective Projection Parameters", padding="10")
    control_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 10))
    control_frame.configure(width=350)
    
    # Right panel for images
    image_frame = ttk.Frame(self.perspective_frame)
    image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Setup perspective controls
    self.setup_perspective_controls(control_frame)
    
    # Setup perspective image display
    self.setup_perspective_image_display(image_frame)
  
  def setup_spherical_tab(self) -> None:
    """Setup the spherical projection tab."""
    # Configure grid weights
    self.spherical_frame.columnconfigure(0, weight=0)  # Controls fixed width
    self.spherical_frame.columnconfigure(1, weight=1)  # Images take remaining space
    self.spherical_frame.rowconfigure(0, weight=1)
    
    # Left panel for controls
    control_frame = ttk.LabelFrame(self.spherical_frame, text="Spherical Projection Parameters", padding="10")
    control_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 10))
    control_frame.configure(width=350)
    
    # Right panel for images
    image_frame = ttk.Frame(self.spherical_frame)
    image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Setup spherical controls
    self.setup_spherical_controls(control_frame)
    
    # Setup spherical image display
    self.setup_spherical_image_display(image_frame)
    
  def setup_perspective_controls(self, parent: ttk.Widget) -> None:
    """Setup perspective projection controls."""
    row = 0
    
    # Image selector
    ttk.Label(parent, text="Source Image Selection", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Image:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    
    # Create combobox for image selection
    self.perspective_image_var = tk.StringVar()
    image_names = [os.path.basename(path) for path in self.available_images] if self.available_images else ["No images found"]
    current_image_name = os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else ""
    
    self.perspective_image_combo = ttk.Combobox(parent, textvariable=self.perspective_image_var, 
                                              values=image_names, state="readonly", width=25)
    self.perspective_image_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
    
    # Set current selection
    if current_image_name in image_names:
      self.perspective_image_combo.set(current_image_name)
    elif image_names and image_names[0] != "No images found":
      self.perspective_image_combo.set(image_names[0])
    
    self.perspective_image_combo.bind('<<ComboboxSelected>>', self.on_perspective_image_change)
    row += 1
    
    # Separator
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    # Output dimensions
    ttk.Label(parent, text="Output Dimensions", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Width:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    width_spinbox = ttk.Spinbox(parent, from_=200, to=2000, increment=50, textvariable=self.perspective_params['output_width'], width=10)
    width_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
    width_spinbox.bind('<Return>', lambda e: self.on_perspective_param_change())
    width_spinbox.bind('<<Increment>>', lambda e: self.on_perspective_param_change())
    width_spinbox.bind('<<Decrement>>', lambda e: self.on_perspective_param_change())
    row += 1
    
    ttk.Label(parent, text="Height:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    height_spinbox = ttk.Spinbox(parent, from_=200, to=2000, increment=50, textvariable=self.perspective_params['output_height'], width=10)
    height_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
    height_spinbox.bind('<Return>', lambda e: self.on_perspective_param_change())
    height_spinbox.bind('<<Increment>>', lambda e: self.on_perspective_param_change())
    height_spinbox.bind('<<Decrement>>', lambda e: self.on_perspective_param_change())
    row += 1
    
    # Rotation controls
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Rotation Controls", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    # Yaw
    ttk.Label(parent, text="Yaw (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    yaw_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.perspective_params['yaw_offset'], orient=tk.HORIZONTAL, length=200)
    yaw_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    yaw_scale.bind('<ButtonRelease-1>', lambda e: self.on_perspective_param_change())
    row += 1
    
    yaw_control_frame = ttk.Frame(parent)
    yaw_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(yaw_control_frame, text="-", width=3, command=lambda: self.adjust_perspective_rotation('yaw', -10)).pack(side=tk.LEFT, padx=(0, 2))
    yaw_entry = ttk.Entry(yaw_control_frame, textvariable=self.perspective_params['yaw_offset'], width=8)
    yaw_entry.pack(side=tk.LEFT, padx=2)
    yaw_entry.bind('<Return>', lambda e: self.on_perspective_param_change())
    ttk.Button(yaw_control_frame, text="+", width=3, command=lambda: self.adjust_perspective_rotation('yaw', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Pitch
    ttk.Label(parent, text="Pitch (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    pitch_scale = ttk.Scale(parent, from_=-90, to=90, variable=self.perspective_params['pitch_offset'], orient=tk.HORIZONTAL, length=200)
    pitch_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    pitch_scale.bind('<ButtonRelease-1>', lambda e: self.on_perspective_param_change())
    row += 1
    
    pitch_control_frame = ttk.Frame(parent)
    pitch_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(pitch_control_frame, text="-", width=3, command=lambda: self.adjust_perspective_rotation('pitch', -10)).pack(side=tk.LEFT, padx=(0, 2))
    pitch_entry = ttk.Entry(pitch_control_frame, textvariable=self.perspective_params['pitch_offset'], width=8)
    pitch_entry.pack(side=tk.LEFT, padx=2)
    pitch_entry.bind('<Return>', lambda e: self.on_perspective_param_change())
    ttk.Button(pitch_control_frame, text="+", width=3, command=lambda: self.adjust_perspective_rotation('pitch', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Roll
    ttk.Label(parent, text="Roll (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    roll_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.perspective_params['roll_offset'], orient=tk.HORIZONTAL, length=200)
    roll_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    roll_scale.bind('<ButtonRelease-1>', lambda e: self.on_perspective_param_change())
    row += 1
    
    roll_control_frame = ttk.Frame(parent)
    roll_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(roll_control_frame, text="-", width=3, command=lambda: self.adjust_perspective_rotation('roll', -10)).pack(side=tk.LEFT, padx=(0, 2))
    roll_entry = ttk.Entry(roll_control_frame, textvariable=self.perspective_params['roll_offset'], width=8)
    roll_entry.pack(side=tk.LEFT, padx=2)
    roll_entry.bind('<Return>', lambda e: self.on_perspective_param_change())
    ttk.Button(roll_control_frame, text="+", width=3, command=lambda: self.adjust_perspective_rotation('roll', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Field of view
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Field of View", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Horizontal FOV (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    fov_scale = ttk.Scale(parent, from_=10, to=175, variable=self.perspective_params['fov_horizontal'], orient=tk.HORIZONTAL, length=200)
    fov_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    fov_scale.bind('<ButtonRelease-1>', lambda e: self.on_perspective_param_change())
    row += 1
    
    fov_control_frame = ttk.Frame(parent)
    fov_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(fov_control_frame, text="-", width=3, command=lambda: self.adjust_perspective_fov(-10)).pack(side=tk.LEFT, padx=(0, 2))
    fov_entry = ttk.Entry(fov_control_frame, textvariable=self.perspective_params['fov_horizontal'], width=8)
    fov_entry.pack(side=tk.LEFT, padx=2)
    fov_entry.bind('<Return>', lambda e: self.on_perspective_param_change())
    ttk.Button(fov_control_frame, text="+", width=3, command=lambda: self.adjust_perspective_fov(10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Virtual camera parameters
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Virtual Camera (0 = auto)", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Virtual fx:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    fx_entry = ttk.Entry(parent, textvariable=self.perspective_params['virtual_fx'], width=10)
    fx_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
    fx_entry.bind('<Return>', lambda e: self.on_perspective_param_change())
    row += 1
    
    ttk.Label(parent, text="Virtual fy:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    fy_entry = ttk.Entry(parent, textvariable=self.perspective_params['virtual_fy'], width=10)
    fy_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
    fy_entry.bind('<Return>', lambda e: self.on_perspective_param_change())
    row += 1
    
    # Projection options
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Projection Options", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    behind_camera_check = ttk.Checkbutton(parent, text="Allow behind camera content", 
                                        variable=self.perspective_params['allow_behind_camera'], 
                                        command=self.on_perspective_param_change)
    behind_camera_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0), pady=2)
    row += 1
    
    # Action buttons
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    button_frame = ttk.Frame(parent)
    button_frame.grid(row=row, column=0, columnspan=2, pady=10)
    
    ttk.Button(button_frame, text="Reset", command=self.reset_perspective_params).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Save Image", command=self.save_perspective_image).pack(side=tk.LEFT, padx=5)
    
    # Presets
    row += 1
    ttk.Label(parent, text="Presets", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
    row += 1
    
    preset_frame = ttk.Frame(parent)
    preset_frame.grid(row=row, column=0, columnspan=2, pady=5)
    
    ttk.Button(preset_frame, text="Wide Angle", command=lambda: self.apply_perspective_preset('wide')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Standard", command=lambda: self.apply_perspective_preset('standard')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Telephoto", command=lambda: self.apply_perspective_preset('telephoto')).pack(side=tk.LEFT, padx=2)
    
    row += 1
    preset_frame2 = ttk.Frame(parent)
    preset_frame2.grid(row=row, column=0, columnspan=2, pady=5)
    
    ttk.Button(preset_frame2, text="Look Up", command=lambda: self.apply_perspective_preset('up')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Look Down", command=lambda: self.apply_perspective_preset('down')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Side View", command=lambda: self.apply_perspective_preset('side')).pack(side=tk.LEFT, padx=2)
    
    # Keyboard shortcuts
    row += 1
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Keyboard Shortcuts", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    shortcuts_frame = ttk.Frame(parent)
    shortcuts_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 0))
    
    # Create shortcuts text with better formatting
    shortcuts_text = tk.Text(
      shortcuts_frame,
      height=6,
      width=40,
      font=('Consolas', 8),
      bg='#f0f0f0',
      fg='#333333',
      wrap=tk.WORD,
      state=tk.DISABLED,
      cursor='arrow',
      relief=tk.FLAT,
      borderwidth=1
    )
    shortcuts_text.pack(fill=tk.BOTH, expand=True)
    
    # Add shortcuts content
    shortcuts_content = """Up Arrow    : Pitch +10 degrees
Down Arrow  : Pitch -10 degrees
Left Arrow  : Yaw -10 degrees
Right Arrow : Yaw +10 degrees
Plus Key    : FOV +10 degrees
Minus Key   : FOV -10 degrees"""
    
    shortcuts_text.configure(state=tk.NORMAL)
    shortcuts_text.insert('1.0', shortcuts_content)
    shortcuts_text.configure(state=tk.DISABLED)
  
  def setup_spherical_controls(self, parent: ttk.Widget) -> None:
    """Setup spherical projection controls."""
    row = 0
    
    # Image selector
    ttk.Label(parent, text="Source Image Selection", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Image:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    
    # Create combobox for image selection
    self.spherical_image_var = tk.StringVar()
    image_names = [os.path.basename(path) for path in self.available_images] if self.available_images else ["No images found"]
    current_image_name = os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else ""
    
    self.spherical_image_combo = ttk.Combobox(parent, textvariable=self.spherical_image_var, 
                                            values=image_names, state="readonly", width=25)
    self.spherical_image_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
    
    # Set current selection
    if current_image_name in image_names:
      self.spherical_image_combo.set(current_image_name)
    elif image_names and image_names[0] != "No images found":
      self.spherical_image_combo.set(image_names[0])
    
    self.spherical_image_combo.bind('<<ComboboxSelected>>', self.on_spherical_image_change)
    row += 1
    
    # Separator
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    # Output dimensions
    ttk.Label(parent, text="Output Dimensions", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Width:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    width_spinbox = ttk.Spinbox(parent, from_=512, to=4096, increment=256, textvariable=self.spherical_params['output_width'], width=10)
    width_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
    width_spinbox.bind('<Return>', lambda e: self.on_spherical_param_change())
    width_spinbox.bind('<<Increment>>', lambda e: self.on_spherical_param_change())
    width_spinbox.bind('<<Decrement>>', lambda e: self.on_spherical_param_change())
    row += 1
    
    ttk.Label(parent, text="Height:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    height_spinbox = ttk.Spinbox(parent, from_=256, to=2048, increment=128, textvariable=self.spherical_params['output_height'], width=10)
    height_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
    height_spinbox.bind('<Return>', lambda e: self.on_spherical_param_change())
    height_spinbox.bind('<<Increment>>', lambda e: self.on_spherical_param_change())
    height_spinbox.bind('<<Decrement>>', lambda e: self.on_spherical_param_change())
    row += 1
    
    # Rotation controls
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Rotation Controls", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    # Yaw
    ttk.Label(parent, text="Yaw (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    yaw_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.spherical_params['yaw_offset'], orient=tk.HORIZONTAL, length=200)
    yaw_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    yaw_scale.bind('<ButtonRelease-1>', lambda e: self.on_spherical_param_change())
    row += 1
    
    yaw_control_frame = ttk.Frame(parent)
    yaw_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(yaw_control_frame, text="-", width=3, command=lambda: self.adjust_spherical_rotation('yaw', -10)).pack(side=tk.LEFT, padx=(0, 2))
    yaw_entry = ttk.Entry(yaw_control_frame, textvariable=self.spherical_params['yaw_offset'], width=8)
    yaw_entry.pack(side=tk.LEFT, padx=2)
    yaw_entry.bind('<Return>', lambda e: self.on_spherical_param_change())
    ttk.Button(yaw_control_frame, text="+", width=3, command=lambda: self.adjust_spherical_rotation('yaw', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Pitch
    ttk.Label(parent, text="Pitch (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    pitch_scale = ttk.Scale(parent, from_=-90, to=90, variable=self.spherical_params['pitch_offset'], orient=tk.HORIZONTAL, length=200)
    pitch_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    pitch_scale.bind('<ButtonRelease-1>', lambda e: self.on_spherical_param_change())
    row += 1
    
    pitch_control_frame = ttk.Frame(parent)
    pitch_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(pitch_control_frame, text="-", width=3, command=lambda: self.adjust_spherical_rotation('pitch', -10)).pack(side=tk.LEFT, padx=(0, 2))
    pitch_entry = ttk.Entry(pitch_control_frame, textvariable=self.spherical_params['pitch_offset'], width=8)
    pitch_entry.pack(side=tk.LEFT, padx=2)
    pitch_entry.bind('<Return>', lambda e: self.on_spherical_param_change())
    ttk.Button(pitch_control_frame, text="+", width=3, command=lambda: self.adjust_spherical_rotation('pitch', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Field of view controls
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Field of View", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    # Horizontal FOV
    ttk.Label(parent, text="Horizontal FOV (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    fov_h_scale = ttk.Scale(parent, from_=60, to=360, variable=self.spherical_params['fov_horizontal'], orient=tk.HORIZONTAL, length=200)
    fov_h_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    fov_h_scale.bind('<ButtonRelease-1>', lambda e: self.on_spherical_param_change())
    row += 1
    
    fov_h_control_frame = ttk.Frame(parent)
    fov_h_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(fov_h_control_frame, text="-", width=3, command=lambda: self.adjust_spherical_fov('horizontal', -30)).pack(side=tk.LEFT, padx=(0, 2))
    fov_h_entry = ttk.Entry(fov_h_control_frame, textvariable=self.spherical_params['fov_horizontal'], width=8)
    fov_h_entry.pack(side=tk.LEFT, padx=2)
    fov_h_entry.bind('<Return>', lambda e: self.on_spherical_param_change())
    ttk.Button(fov_h_control_frame, text="+", width=3, command=lambda: self.adjust_spherical_fov('horizontal', 30)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Vertical FOV
    ttk.Label(parent, text="Vertical FOV (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    fov_v_scale = ttk.Scale(parent, from_=30, to=180, variable=self.spherical_params['fov_vertical'], orient=tk.HORIZONTAL, length=200)
    fov_v_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    fov_v_scale.bind('<ButtonRelease-1>', lambda e: self.on_spherical_param_change())
    row += 1
    
    fov_v_control_frame = ttk.Frame(parent)
    fov_v_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(fov_v_control_frame, text="-", width=3, command=lambda: self.adjust_spherical_fov('vertical', -15)).pack(side=tk.LEFT, padx=(0, 2))
    fov_v_entry = ttk.Entry(fov_v_control_frame, textvariable=self.spherical_params['fov_vertical'], width=8)
    fov_v_entry.pack(side=tk.LEFT, padx=2)
    fov_v_entry.bind('<Return>', lambda e: self.on_spherical_param_change())
    ttk.Button(fov_v_control_frame, text="+", width=3, command=lambda: self.adjust_spherical_fov('vertical', 15)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Allow behind camera checkbox
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    behind_camera_check = ttk.Checkbutton(parent, text="Allow behind camera content", 
                                        variable=self.spherical_params['allow_behind_camera'], 
                                        command=self.on_spherical_param_change)
    behind_camera_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0), pady=2)
    row += 1
    
    # Action buttons
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    button_frame = ttk.Frame(parent)
    button_frame.grid(row=row, column=0, columnspan=2, pady=10)
    
    ttk.Button(button_frame, text="Reset", command=self.reset_spherical_params).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Save Image", command=self.save_spherical_image).pack(side=tk.LEFT, padx=5)
    
    # Presets
    row += 1
    ttk.Label(parent, text="Presets", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
    row += 1
    
    preset_frame = ttk.Frame(parent)
    preset_frame.grid(row=row, column=0, columnspan=2, pady=5)
    
    ttk.Button(preset_frame, text="Full Sphere", command=lambda: self.apply_spherical_preset('full_sphere')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Hemisphere", command=lambda: self.apply_spherical_preset('hemisphere')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Panoramic", command=lambda: self.apply_spherical_preset('panoramic')).pack(side=tk.LEFT, padx=2)
    
    row += 1
    preset_frame2 = ttk.Frame(parent)
    preset_frame2.grid(row=row, column=0, columnspan=2, pady=5)
    
    ttk.Button(preset_frame2, text="Equatorial", command=lambda: self.apply_spherical_preset('equatorial')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Look Up", command=lambda: self.apply_spherical_preset('look_up')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Look Down", command=lambda: self.apply_spherical_preset('look_down')).pack(side=tk.LEFT, padx=2)
    
    # Keyboard shortcuts
    row += 1
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Keyboard Shortcuts", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    shortcuts_frame = ttk.Frame(parent)
    shortcuts_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 0))
    
    # Create shortcuts text with better formatting
    shortcuts_text = tk.Text(
      shortcuts_frame,
      height=6,
      width=40,
      font=('Consolas', 8),
      bg='#f0f0f0',
      fg='#333333',
      wrap=tk.WORD,
      state=tk.DISABLED,
      cursor='arrow',
      relief=tk.FLAT,
      borderwidth=1
    )
    shortcuts_text.pack(fill=tk.BOTH, expand=True)
    
    # Add shortcuts content
    shortcuts_content = """Up Arrow    : Pitch +10 degrees
Down Arrow  : Pitch -10 degrees
Left Arrow  : Yaw -10 degrees
Right Arrow : Yaw +10 degrees
Plus Key    : FOV +10 degrees
Minus Key   : FOV -10 degrees"""
    
    shortcuts_text.configure(state=tk.NORMAL)
    shortcuts_text.insert('1.0', shortcuts_content)
    shortcuts_text.configure(state=tk.DISABLED)
  
  def setup_perspective_image_display(self, parent: ttk.Widget) -> None:
    """Setup perspective projection image display."""
    # Image display frame
    display_frame = ttk.Frame(parent)
    display_frame.pack(fill=tk.BOTH, expand=True)
    
    # Configure grid weights for side-by-side layout
    display_frame.columnconfigure(0, weight=1)
    display_frame.columnconfigure(1, weight=1)
    display_frame.rowconfigure(0, weight=3)  # Images take most space
    display_frame.rowconfigure(1, weight=1)  # Terminal takes remaining space
    
    # Left side - Original image
    left_frame = ttk.Frame(display_frame)
    left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
    
    ttk.Label(left_frame, text="Original Fisheye Image", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
    
    self.perspective_original_label = ttk.Label(left_frame)
    self.perspective_original_label.pack()
    
    # Right side - Projected image
    right_frame = ttk.Frame(display_frame)
    right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
    
    ttk.Label(right_frame, text="Perspective Projection Result", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
    
    self.perspective_projected_label = ttk.Label(right_frame)
    self.perspective_projected_label.pack()
    
    # Terminal-like output area
    terminal_frame = ttk.LabelFrame(display_frame, text="Program Output", padding="5")
    terminal_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
    terminal_frame.columnconfigure(0, weight=1)
    terminal_frame.rowconfigure(0, weight=1)
    
    self.perspective_terminal_text = tk.Text(
      terminal_frame,
      bg='black',
      fg='#00ff00',
      font=('Consolas', 9),
      wrap=tk.WORD,
      state=tk.DISABLED,
      cursor='arrow'
    )
    
    terminal_scrollbar = ttk.Scrollbar(terminal_frame, orient=tk.VERTICAL, command=self.perspective_terminal_text.yview)
    self.perspective_terminal_text.configure(yscrollcommand=terminal_scrollbar.set)
    
    self.perspective_terminal_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    terminal_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
  
  def setup_spherical_image_display(self, parent: ttk.Widget) -> None:
    """Setup spherical projection image display."""
    # Image display frame
    display_frame = ttk.Frame(parent)
    display_frame.pack(fill=tk.BOTH, expand=True)
    
    # Configure grid weights for side-by-side layout
    display_frame.columnconfigure(0, weight=1)
    display_frame.columnconfigure(1, weight=1)
    display_frame.rowconfigure(0, weight=3)  # Images take most space
    display_frame.rowconfigure(1, weight=1)  # Terminal takes remaining space
    
    # Left side - Original image
    left_frame = ttk.Frame(display_frame)
    left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
    
    ttk.Label(left_frame, text="Original Fisheye Image", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
    
    self.spherical_original_label = ttk.Label(left_frame)
    self.spherical_original_label.pack()
    
    # Right side - Projected image
    right_frame = ttk.Frame(display_frame)
    right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
    
    ttk.Label(right_frame, text="Spherical Projection Result", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
    
    self.spherical_projected_label = ttk.Label(right_frame)
    self.spherical_projected_label.pack()
    
    # Terminal-like output area
    terminal_frame = ttk.LabelFrame(display_frame, text="Program Output", padding="5")
    terminal_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
    terminal_frame.columnconfigure(0, weight=1)
    terminal_frame.rowconfigure(0, weight=1)
    
    self.spherical_terminal_text = tk.Text(
      terminal_frame,
      bg='black',
      fg='#00ff00',
      font=('Consolas', 9),
      wrap=tk.WORD,
      state=tk.DISABLED,
      cursor='arrow'
    )
    
    terminal_scrollbar = ttk.Scrollbar(terminal_frame, orient=tk.VERTICAL, command=self.spherical_terminal_text.yview)
    self.spherical_terminal_text.configure(yscrollcommand=terminal_scrollbar.set)
    
    self.spherical_terminal_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    terminal_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
  
  def on_tab_changed(self, event) -> None:
    """Handle tab change events."""
    selected_tab = self.notebook.select()
    tab_text = self.notebook.tab(selected_tab, "text")
    
    if tab_text == "Perspective Projection":
      self.log_perspective_message("Switched to Perspective Projection")
      self.update_perspective_images()
    elif tab_text == "Spherical Projection":
      self.log_spherical_message("Switched to Spherical Projection")
      self.update_spherical_images()
    
    # Ensure focus stays on root window for keyboard controls
    self.root.focus_set()
  
  def on_perspective_param_change(self) -> None:
    """Handle perspective parameter changes with debouncing."""
    if hasattr(self, 'perspective_pending_update_id') and self.perspective_pending_update_id:
      self.root.after_cancel(self.perspective_pending_update_id)
    
    self.perspective_pending_update_id = self.root.after(100, self.update_perspective_images)
  
  def on_perspective_image_change(self, event) -> None:
    """Handle perspective image selection change."""
    selected_name = self.perspective_image_var.get()
    if selected_name and selected_name != "No images found":
      # Find the full path for the selected image
      for img_path in self.available_images:
        if os.path.basename(img_path) == selected_name:
          self.load_new_image(img_path)
          self.log_perspective_message(f"Switched to image: {selected_name}")
          break

  def on_spherical_image_change(self, event) -> None:
    """Handle spherical image selection change."""
    selected_name = self.spherical_image_var.get()
    if selected_name and selected_name != "No images found":
      # Find the full path for the selected image
      for img_path in self.available_images:
        if os.path.basename(img_path) == selected_name:
          self.load_new_image(img_path)
          self.log_spherical_message(f"Switched to image: {selected_name}")
          break

  def load_new_image(self, image_path: str) -> None:
    """Load a new fisheye image and update projections."""
    try:
      new_img = cv2.imread(image_path)
      if new_img is None:
        raise ValueError(f"Could not load image: {image_path}")
      
      self.fisheye_img = new_img
      self.current_image_path = image_path
      
      # Update both combo boxes to show the same selection
      image_name = os.path.basename(image_path)
      if hasattr(self, 'perspective_image_combo'):
        self.perspective_image_combo.set(image_name)
      if hasattr(self, 'spherical_image_combo'):
        self.spherical_image_combo.set(image_name)
      
      # Clear cache since we have a new image
      if hasattr(self, 'shared_cache'):
        self.shared_cache.clear()
      
      # Update both projections with the new image
      self.update_perspective_images()
      self.update_spherical_images()
      
    except Exception as e:
      error_msg = f"Failed to load image {image_path}: {e}"
      self.log_perspective_message(error_msg)
      self.log_spherical_message(error_msg)
      messagebox.showerror("Image Load Error", error_msg)

  def on_spherical_param_change(self) -> None:
    """Handle spherical parameter changes with debouncing."""
    if hasattr(self, 'spherical_pending_update_id') and self.spherical_pending_update_id:
      self.root.after_cancel(self.spherical_pending_update_id)
    
    self.spherical_pending_update_id = self.root.after(100, self.update_spherical_images)
  
  def update_images(self) -> None:
    """Initial image update for both tabs."""
    self.update_perspective_images()
    self.update_spherical_images()
    
    # Initialize terminal messages
    self.log_perspective_message("Fisheye Perspective Projection Tool initialized")
    self.log_perspective_message(f"Shared cache initialized with 256MB LRU limit")
    self.log_perspective_message("Ready for processing...")
    self.log_spherical_message("Fisheye Spherical Projection Tool initialized")
    self.log_spherical_message(f"Shared cache initialized with 256MB LRU limit")
    self.log_spherical_message("Ready for processing...")
  
  def update_perspective_images(self) -> None:
    """Update perspective projection images."""
    self.update_original_image(self.perspective_original_label)
    
    if self.perspective_processing_thread and self.perspective_processing_thread.is_alive():
      try:
        self.perspective_processing_queue.put_nowait('update')
      except queue.Full:
        pass
    else:
      self.perspective_processing_thread = threading.Thread(target=self.process_perspective_projection)
      self.perspective_processing_thread.daemon = True
      self.perspective_processing_thread.start()
  
  def update_spherical_images(self) -> None:
    """Update spherical projection images."""
    self.update_original_image(self.spherical_original_label)
    
    if self.spherical_processing_thread and self.spherical_processing_thread.is_alive():
      try:
        self.spherical_processing_queue.put_nowait('update')
      except queue.Full:
        pass
    else:
      self.spherical_processing_thread = threading.Thread(target=self.process_spherical_projection)
      self.spherical_processing_thread.daemon = True
      self.spherical_processing_thread.start()
  
  def update_original_image(self, label_widget) -> None:
    """Update original fisheye image display."""
    display_size = 600
    
    img_resized = cv2.resize(self.fisheye_img, (display_size, display_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    label_widget.configure(image=img_tk)
    label_widget.image = img_tk  # Keep a reference
  
  def process_perspective_projection(self) -> None:
    """Process perspective projection in background thread."""
    try:
      self.log_perspective_message("Starting perspective projection processing...")
      
      params = {k: v.get() for k, v in self.perspective_params.items()}
      
      self.log_perspective_message(f"Parameters: Yaw={params['yaw_offset']:.1f}°, Pitch={params['pitch_offset']:.1f}°, Roll={params['roll_offset']:.1f}°, FOV={params['fov_horizontal']:.1f}°")
      
      virtual_fx = params['virtual_fx'] if params['virtual_fx'] > 0 else None
      virtual_fy = params['virtual_fy'] if params['virtual_fy'] > 0 else None
      
      projected_img = self.perspective_projector.project(
        self.fisheye_img,
        output_width=params['output_width'],
        output_height=params['output_height'],
        yaw_offset=params['yaw_offset'],
        pitch_offset=params['pitch_offset'],
        roll_offset=params['roll_offset'],
        fov_horizontal=params['fov_horizontal'],
        virtual_fx=virtual_fx,
        virtual_fy=virtual_fy,
        allow_behind_camera=params['allow_behind_camera']
      )
      
      try:
        self.perspective_processing_queue.get_nowait()
        self.log_perspective_message("New parameter change detected, restarting processing...")
        self.process_perspective_projection()
        return
      except queue.Empty:
        pass
      
      self.perspective_result_queue.put(projected_img)
      
    except Exception as e:
      error_msg = f"Processing error: {e}"
      self.log_perspective_message(error_msg)
      self.perspective_result_queue.put(f"Error: {e}")
  
  def process_spherical_projection(self) -> None:
    """Process spherical projection in background thread."""
    try:
      self.log_spherical_message("Starting spherical projection processing...")
      
      params = {k: v.get() for k, v in self.spherical_params.items()}
      
      self.log_spherical_message(f"Parameters: Yaw={params['yaw_offset']:.1f}°, Pitch={params['pitch_offset']:.1f}°")
      self.log_spherical_message(f"FOV: H={params['fov_horizontal']:.1f}°, V={params['fov_vertical']:.1f}°")
      
      projected_img = self.spherical_projector.project(
        self.fisheye_img,
        output_width=params['output_width'],
        output_height=params['output_height'],
        yaw_offset=params['yaw_offset'],
        pitch_offset=params['pitch_offset'],
        fov_horizontal=params['fov_horizontal'],
        fov_vertical=params['fov_vertical'],
        allow_behind_camera=params['allow_behind_camera']
      )
      
      try:
        self.spherical_processing_queue.get_nowait()
        self.log_spherical_message("New parameter change detected, restarting processing...")
        self.process_spherical_projection()
        return
      except queue.Empty:
        pass
      
      self.spherical_result_queue.put(projected_img)
      
    except Exception as e:
      error_msg = f"Processing error: {e}"
      self.log_spherical_message(error_msg)
      self.spherical_result_queue.put(f"Error: {e}")
  
  def check_results(self) -> None:
    """Check for processing results from both projections."""
    # Check perspective results
    try:
      result = self.perspective_result_queue.get_nowait()
      if isinstance(result, str) and result.startswith("Error"):
        self.log_perspective_message(result)
      else:
        self.update_perspective_projected_image(result)
        self.log_perspective_message("Perspective projection completed successfully")
    except queue.Empty:
      pass
    
    # Check spherical results
    try:
      result = self.spherical_result_queue.get_nowait()
      if isinstance(result, str) and result.startswith("Error"):
        self.log_spherical_message(result)
      else:
        self.update_spherical_projected_image(result)
        self.log_spherical_message("Spherical projection completed successfully")
    except queue.Empty:
      pass
    
    # Schedule next check
    self.root.after(30, self.check_results)
  
  def update_perspective_projected_image(self, projected_img: np.ndarray) -> None:
    """Update perspective projected image display."""
    display_size = 600
    h, w = projected_img.shape[:2]
    
    if w > h:
      new_w = display_size
      new_h = int(h * display_size / w)
    else:
      new_h = display_size
      new_w = int(w * display_size / h)
    
    img_resized = cv2.resize(projected_img, (new_w, new_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    self.perspective_projected_label.configure(image=img_tk)
    self.perspective_projected_label.image = img_tk
    
    self.last_perspective_projected_img = projected_img
  
  def update_spherical_projected_image(self, projected_img: np.ndarray) -> None:
    """Update spherical projected image display."""
    display_width = 600
    h, w = projected_img.shape[:2]
    
    aspect_ratio = w / h
    if aspect_ratio > 1:
      new_w = display_width
      new_h = int(display_width / aspect_ratio)
    else:
      new_h = display_width
      new_w = int(display_width * aspect_ratio)
    
    if new_h < 200:
      new_h = 200
      new_w = int(200 * aspect_ratio)
    if new_w < 200:
      new_w = 200
      new_h = int(200 / aspect_ratio)
    
    img_resized = cv2.resize(projected_img, (new_w, new_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    self.spherical_projected_label.configure(image=img_tk)
    self.spherical_projected_label.image = img_tk
    
    self.last_spherical_projected_img = projected_img
  
  def log_perspective_message(self, message: str) -> None:
    """Add message to perspective terminal output."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    
    self.perspective_terminal_text.configure(state=tk.NORMAL)
    self.perspective_terminal_text.insert(tk.END, formatted_message)
    self.perspective_terminal_text.see(tk.END)
    self.perspective_terminal_text.configure(state=tk.DISABLED)
    self.perspective_terminal_text.update_idletasks()
  
  def log_spherical_message(self, message: str) -> None:
    """Add message to spherical terminal output."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    
    self.spherical_terminal_text.configure(state=tk.NORMAL)
    self.spherical_terminal_text.insert(tk.END, formatted_message)
    self.spherical_terminal_text.see(tk.END)
    self.spherical_terminal_text.configure(state=tk.DISABLED)
    self.spherical_terminal_text.update_idletasks()
  
  def reset_perspective_params(self) -> None:
    """Reset perspective parameters to defaults."""
    self.log_perspective_message("Resetting perspective parameters to default values")
    self.perspective_params['output_width'].set(800)
    self.perspective_params['output_height'].set(800)
    self.perspective_params['yaw_offset'].set(0.0)
    self.perspective_params['pitch_offset'].set(0.0)
    self.perspective_params['roll_offset'].set(0.0)
    self.perspective_params['fov_horizontal'].set(90.0)
    self.perspective_params['virtual_fx'].set(0.0)
    self.perspective_params['virtual_fy'].set(0.0)
    self.perspective_params['allow_behind_camera'].set(True)
    self.update_perspective_images()
  
  def reset_spherical_params(self) -> None:
    """Reset spherical parameters to defaults."""
    self.log_spherical_message("Resetting spherical parameters to default values")
    self.spherical_params['output_width'].set(2048)
    self.spherical_params['output_height'].set(1024)
    self.spherical_params['yaw_offset'].set(0.0)
    self.spherical_params['pitch_offset'].set(0.0)
    self.spherical_params['fov_horizontal'].set(360.0)
    self.spherical_params['fov_vertical'].set(180.0)
    self.spherical_params['allow_behind_camera'].set(True)
    self.update_spherical_images()
  
  def apply_spherical_preset(self, preset_type: str) -> None:
    """Apply spherical projection presets."""
    self.log_spherical_message(f"Applying '{preset_type}' preset")
    if preset_type == 'full_sphere':
      self.spherical_params['fov_horizontal'].set(360.0)
      self.spherical_params['fov_vertical'].set(180.0)
      self.spherical_params['yaw_offset'].set(0.0)
      self.spherical_params['pitch_offset'].set(0.0)
      self.spherical_params['output_width'].set(2048)
      self.spherical_params['output_height'].set(1024)
    elif preset_type == 'hemisphere':
      self.spherical_params['fov_horizontal'].set(360.0)
      self.spherical_params['fov_vertical'].set(90.0)
      self.spherical_params['yaw_offset'].set(0.0)
      self.spherical_params['pitch_offset'].set(0.0)
      self.spherical_params['output_width'].set(2048)
      self.spherical_params['output_height'].set(512)
    elif preset_type == 'panoramic':
      self.spherical_params['fov_horizontal'].set(360.0)
      self.spherical_params['fov_vertical'].set(120.0)
      self.spherical_params['yaw_offset'].set(0.0)
      self.spherical_params['pitch_offset'].set(0.0)
      self.spherical_params['output_width'].set(3600)
      self.spherical_params['output_height'].set(1200)
    elif preset_type == 'equatorial':
      self.spherical_params['fov_horizontal'].set(360.0)
      self.spherical_params['fov_vertical'].set(60.0)
      self.spherical_params['yaw_offset'].set(0.0)
      self.spherical_params['pitch_offset'].set(0.0)
      self.spherical_params['output_width'].set(3600)
      self.spherical_params['output_height'].set(600)
    elif preset_type == 'look_up':
      self.spherical_params['fov_horizontal'].set(180.0)
      self.spherical_params['fov_vertical'].set(90.0)
      self.spherical_params['yaw_offset'].set(0.0)
      self.spherical_params['pitch_offset'].set(45.0)
      self.spherical_params['output_width'].set(1800)
      self.spherical_params['output_height'].set(900)
    elif preset_type == 'look_down':
      self.spherical_params['fov_horizontal'].set(180.0)
      self.spherical_params['fov_vertical'].set(90.0)
      self.spherical_params['yaw_offset'].set(0.0)
      self.spherical_params['pitch_offset'].set(-45.0)
      self.spherical_params['output_width'].set(1800)
      self.spherical_params['output_height'].set(900)
    
    self.update_spherical_images()
  
  def adjust_spherical_fov(self, axis: str, delta: float) -> None:
    """Adjust spherical field of view value by delta degrees with proper range clamping."""
    if axis == 'horizontal':
      current = self.spherical_params['fov_horizontal'].get()
      new_value = max(60, min(360, current + delta))  # Clamp to 60 to 360 range
      self.spherical_params['fov_horizontal'].set(new_value)
    elif axis == 'vertical':
      current = self.spherical_params['fov_vertical'].get()
      new_value = max(30, min(180, current + delta))  # Clamp to 30 to 180 range
      self.spherical_params['fov_vertical'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_spherical_param_change()
  
  def save_perspective_image(self) -> None:
    """Save perspective projected image."""
    if hasattr(self, 'last_perspective_projected_img'):
      filename = f"output/perspective/perspective_yaw{self.perspective_params['yaw_offset'].get():.0f}_pitch{self.perspective_params['pitch_offset'].get():.0f}_roll{self.perspective_params['roll_offset'].get():.0f}_fov{self.perspective_params['fov_horizontal'].get():.0f}.jpg"
      cv2.imwrite(filename, self.last_perspective_projected_img)
      self.log_perspective_message(f"Image saved successfully: {filename}")
      messagebox.showinfo("Success", f"Perspective image saved as {filename}")
    else:
      self.log_perspective_message("Save failed: No projected image available")
      messagebox.showwarning("Warning", "No perspective projected image to save")
  
  def save_spherical_image(self) -> None:
    """Save spherical projected image."""
    if hasattr(self, 'last_spherical_projected_img'):
      filename = f"output/spherical/spherical_yaw{self.spherical_params['yaw_offset'].get():.0f}_pitch{self.spherical_params['pitch_offset'].get():.0f}_fovh{self.spherical_params['fov_horizontal'].get():.0f}_fovv{self.spherical_params['fov_vertical'].get():.0f}.jpg"
      cv2.imwrite(filename, self.last_spherical_projected_img)
      self.log_spherical_message(f"Image saved successfully: {filename}")
      messagebox.showinfo("Success", f"Spherical image saved as {filename}")
    else:
      self.log_spherical_message("Save failed: No projected image available")
      messagebox.showwarning("Warning", "No spherical projected image to save")
  
  def adjust_perspective_rotation(self, axis: str, delta: float) -> None:
    """Adjust perspective rotation value by delta degrees with proper range clamping."""
    if axis == 'yaw':
      current = self.perspective_params['yaw_offset'].get()
      new_value = current + delta
      # Clamp to -180 to 180 range
      if new_value > 180:
        new_value -= 360
      elif new_value < -180:
        new_value += 360
      self.perspective_params['yaw_offset'].set(new_value)
    elif axis == 'pitch':
      current = self.perspective_params['pitch_offset'].get()
      new_value = max(-90, min(90, current + delta))  # Clamp to -90 to 90 range
      self.perspective_params['pitch_offset'].set(new_value)
    elif axis == 'roll':
      current = self.perspective_params['roll_offset'].get()
      new_value = current + delta
      # Clamp to -180 to 180 range
      if new_value > 180:
        new_value -= 360
      elif new_value < -180:
        new_value += 360
      self.perspective_params['roll_offset'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_perspective_param_change()
  
  def adjust_perspective_fov(self, delta: float) -> None:
    """Adjust perspective field of view value by delta degrees with proper range clamping."""
    current = self.perspective_params['fov_horizontal'].get()
    new_value = max(10, min(175, current + delta))  # Clamp to 10 to 175 range
    self.perspective_params['fov_horizontal'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_perspective_param_change()
  
  def apply_perspective_preset(self, preset_type: str) -> None:
    """Apply perspective projection presets."""
    self.log_perspective_message(f"Applying '{preset_type}' preset")
    if preset_type == 'wide':
      self.perspective_params['fov_horizontal'].set(120.0)
      self.perspective_params['yaw_offset'].set(0.0)
      self.perspective_params['pitch_offset'].set(0.0)
      self.perspective_params['roll_offset'].set(0.0)
    elif preset_type == 'standard':
      self.perspective_params['fov_horizontal'].set(60.0)
      self.perspective_params['yaw_offset'].set(0.0)
      self.perspective_params['pitch_offset'].set(0.0)
      self.perspective_params['roll_offset'].set(0.0)
    elif preset_type == 'telephoto':
      self.perspective_params['fov_horizontal'].set(30.0)
      self.perspective_params['yaw_offset'].set(0.0)
      self.perspective_params['pitch_offset'].set(0.0)
      self.perspective_params['roll_offset'].set(0.0)
    elif preset_type == 'up':
      self.perspective_params['pitch_offset'].set(45.0)
      self.perspective_params['yaw_offset'].set(0.0)
      self.perspective_params['roll_offset'].set(0.0)
    elif preset_type == 'down':
      self.perspective_params['pitch_offset'].set(-45.0)
      self.perspective_params['yaw_offset'].set(0.0)
      self.perspective_params['roll_offset'].set(0.0)
    elif preset_type == 'side':
      self.perspective_params['yaw_offset'].set(90.0)
      self.perspective_params['pitch_offset'].set(0.0)
      self.perspective_params['roll_offset'].set(0.0)
    
    self.update_perspective_images()
  
  def adjust_spherical_rotation(self, axis: str, delta: float) -> None:
    """Adjust spherical rotation value by delta degrees with proper range clamping."""
    if axis == 'yaw':
      current = self.spherical_params['yaw_offset'].get()
      new_value = current + delta
      # Clamp to -180 to 180 range
      if new_value > 180:
        new_value -= 360
      elif new_value < -180:
        new_value += 360
      self.spherical_params['yaw_offset'].set(new_value)
    elif axis == 'pitch':
      current = self.spherical_params['pitch_offset'].get()
      new_value = max(-90, min(90, current + delta))  # Clamp to -90 to 90 range
      self.spherical_params['pitch_offset'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_spherical_param_change()
  
  def setup_keyboard_controls(self) -> None:
    """Setup keyboard controls for arrow key navigation."""
    # Bind keyboard events to the root window
    self.root.bind('<Key>', self.on_key_press)
    
    # Make the root window focusable so it can receive key events
    self.root.focus_set()
    
    # Log keyboard control setup
    self.log_perspective_message("Keyboard controls enabled: Arrow keys adjust pitch/yaw, +/- keys adjust FOV by 10°")
    self.log_spherical_message("Keyboard controls enabled: Arrow keys adjust pitch/yaw, +/- keys adjust FOV by 10°")
  
  def on_key_press(self, event) -> None:
    """Handle keyboard events for arrow key controls."""
    # Get current active tab
    selected_tab = self.notebook.select()
    tab_text = self.notebook.tab(selected_tab, "text")
    
    # Determine which parameter set to use based on active tab
    if tab_text == "Perspective Projection":
      params = self.perspective_params
      log_func = self.log_perspective_message
      update_func = self.on_perspective_param_change
    elif tab_text == "Spherical Projection":
      params = self.spherical_params
      log_func = self.log_spherical_message
      update_func = self.on_spherical_param_change
    else:
      return
    
    # Handle arrow key presses
    if event.keysym == 'Up':
      # Up arrow increases pitch by 10 degrees
      current_pitch = params['pitch_offset'].get()
      new_pitch = max(-90, min(90, current_pitch + 10))
      params['pitch_offset'].set(new_pitch)
      log_func(f"Keyboard: Pitch increased to {new_pitch:.1f}° (Up arrow)")
      update_func()
      
    elif event.keysym == 'Down':
      # Down arrow decreases pitch by 10 degrees
      current_pitch = params['pitch_offset'].get()
      new_pitch = max(-90, min(90, current_pitch - 10))
      params['pitch_offset'].set(new_pitch)
      log_func(f"Keyboard: Pitch decreased to {new_pitch:.1f}° (Down arrow)")
      update_func()
      
    elif event.keysym == 'Left':
      # Left arrow decreases yaw by 10 degrees
      current_yaw = params['yaw_offset'].get()
      new_yaw = current_yaw - 10
      # Wrap around -180 to 180 range
      if new_yaw < -180:
        new_yaw += 360
      params['yaw_offset'].set(new_yaw)
      log_func(f"Keyboard: Yaw decreased to {new_yaw:.1f}° (Left arrow)")
      update_func()
      
    elif event.keysym == 'Right':
      # Right arrow increases yaw by 10 degrees
      current_yaw = params['yaw_offset'].get()
      new_yaw = current_yaw + 10
      # Wrap around -180 to 180 range
      if new_yaw > 180:
        new_yaw -= 360
      params['yaw_offset'].set(new_yaw)
      log_func(f"Keyboard: Yaw increased to {new_yaw:.1f}° (Right arrow)")
      update_func()
      
    elif event.keysym == 'plus' or event.keysym == 'equal':
      # Plus key increases horizontal FOV by 10 degrees
      current_fov = params['fov_horizontal'].get()
      if tab_text == "Perspective Projection":
        new_fov = max(10, min(175, current_fov + 10))  # Perspective FOV range
      else:  # Spherical projection
        new_fov = max(60, min(360, current_fov + 10))  # Spherical FOV range
      params['fov_horizontal'].set(new_fov)
      log_func(f"Keyboard: Horizontal FOV increased to {new_fov:.1f}° (+ key)")
      update_func()
      
    elif event.keysym == 'minus':
      # Minus key decreases horizontal FOV by 10 degrees
      current_fov = params['fov_horizontal'].get()
      if tab_text == "Perspective Projection":
        new_fov = max(10, min(175, current_fov - 10))  # Perspective FOV range
      else:  # Spherical projection
        new_fov = max(60, min(360, current_fov - 10))  # Spherical FOV range
      params['fov_horizontal'].set(new_fov)
      log_func(f"Keyboard: Horizontal FOV decreased to {new_fov:.1f}° (- key)")
      update_func()

def main() -> None:
  root = tk.Tk()
  app = FisheyeProjectionGUI(root)
  root.mainloop()

if __name__ == "__main__":
  main()
