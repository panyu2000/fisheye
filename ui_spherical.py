import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from camera_params import parse_camera_params
from spherical_projection import SphericalProjection
import threading
import queue
from typing import Dict, Union, Optional

class SphericalFisheyeUI:
  def __init__(self, root: tk.Tk) -> None:
    self.root = root
    self.root.title("Fisheye Spherical Projection Tool")
    self.root.geometry("1600x1000")
    
    # Load camera parameters and fisheye image
    try:
      self.camera_params = parse_camera_params("camera_intrinsics.yaml")
      self.fisheye_img = cv2.imread("fisheye_img.jpg")
      if self.fisheye_img is None:
        raise FileNotFoundError("Could not load fisheye_img.jpg")
        
      # Get image dimensions for validation
      img_height, img_width = self.fisheye_img.shape[:2]
      
      # Create SphericalProjection instance with caching capabilities
      # Use vectorized=True by default for fast performance
      self.projector = SphericalProjection(self.camera_params, input_image_size=(img_width, img_height), use_vectorized=True)
      
    except Exception as e:
      messagebox.showerror("Error", f"Failed to load files: {e}")
      return
    
    # Variables for spherical projection parameters
    self.params = {
      'output_width': tk.IntVar(value=2048),
      'output_height': tk.IntVar(value=1024),
      'yaw_offset': tk.DoubleVar(value=0.0),
      'pitch_offset': tk.DoubleVar(value=0.0),
      'fov_horizontal': tk.DoubleVar(value=360.0),
      'fov_vertical': tk.DoubleVar(value=180.0),
      'allow_behind_camera': tk.BooleanVar(value=True)
    }
    
    # Threading for image processing
    self.processing_queue = queue.Queue()
    self.result_queue = queue.Queue()
    self.processing_thread = None
    self.update_pending = False
    self.text_update_pending = False
    self.last_update_id = None
    self.pending_update_id = None  # For debouncing updates
    
    self.setup_ui()
    self.update_images()
    
    # Start periodic check for results
    self.root.after(100, self.check_results)
    
  def setup_ui(self) -> None:
    # Main container
    main_frame = ttk.Frame(self.root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights - give more space to images
    self.root.columnconfigure(0, weight=1)
    self.root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=0)  # Controls fixed width
    main_frame.columnconfigure(1, weight=1)  # Images take remaining space
    main_frame.rowconfigure(0, weight=1)
    
    # Left panel for controls - fixed width
    control_frame = ttk.LabelFrame(main_frame, text="Spherical Projection Parameters", padding="10")
    control_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 10))
    control_frame.configure(width=350)  # Fixed width for controls
    
    # Right panel for images - takes remaining space
    image_frame = ttk.Frame(main_frame)
    image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Setup control widgets
    self.setup_controls(control_frame)
    
    # Setup image display
    self.setup_image_display(image_frame)
    
  def setup_controls(self, parent: ttk.Widget) -> None:
    row = 0
    
    # Output dimensions
    ttk.Label(parent, text="Output Dimensions", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Width:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    width_spinbox = ttk.Spinbox(parent, from_=512, to=4096, increment=256, textvariable=self.params['output_width'], width=10)
    width_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
    width_spinbox.bind('<Return>', self.on_param_change)
    width_spinbox.bind('<<Increment>>', self.on_param_change)
    width_spinbox.bind('<<Decrement>>', self.on_param_change)
    row += 1
    
    ttk.Label(parent, text="Height:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    height_spinbox = ttk.Spinbox(parent, from_=256, to=2048, increment=128, textvariable=self.params['output_height'], width=10)
    height_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
    height_spinbox.bind('<Return>', self.on_param_change)
    height_spinbox.bind('<<Increment>>', self.on_param_change)
    height_spinbox.bind('<<Decrement>>', self.on_param_change)
    row += 1
    
    # Rotation controls
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Rotation Controls", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    # Yaw
    ttk.Label(parent, text="Yaw (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    yaw_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.params['yaw_offset'], orient=tk.HORIZONTAL, length=200)
    yaw_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    yaw_scale.bind('<ButtonRelease-1>', self.on_param_change)
    row += 1
    
    yaw_control_frame = ttk.Frame(parent)
    yaw_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(yaw_control_frame, text="-", width=3, command=lambda: self.adjust_rotation('yaw', -10)).pack(side=tk.LEFT, padx=(0, 2))
    yaw_entry = ttk.Entry(yaw_control_frame, textvariable=self.params['yaw_offset'], width=8)
    yaw_entry.pack(side=tk.LEFT, padx=2)
    yaw_entry.bind('<Return>', self.on_param_change)
    ttk.Button(yaw_control_frame, text="+", width=3, command=lambda: self.adjust_rotation('yaw', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Pitch
    ttk.Label(parent, text="Pitch (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    pitch_scale = ttk.Scale(parent, from_=-90, to=90, variable=self.params['pitch_offset'], orient=tk.HORIZONTAL, length=200)
    pitch_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    pitch_scale.bind('<ButtonRelease-1>', self.on_param_change)
    row += 1
    
    pitch_control_frame = ttk.Frame(parent)
    pitch_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(pitch_control_frame, text="-", width=3, command=lambda: self.adjust_rotation('pitch', -10)).pack(side=tk.LEFT, padx=(0, 2))
    pitch_entry = ttk.Entry(pitch_control_frame, textvariable=self.params['pitch_offset'], width=8)
    pitch_entry.pack(side=tk.LEFT, padx=2)
    pitch_entry.bind('<Return>', self.on_param_change)
    ttk.Button(pitch_control_frame, text="+", width=3, command=lambda: self.adjust_rotation('pitch', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Field of view controls
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Field of View", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    # Horizontal FOV
    ttk.Label(parent, text="Horizontal FOV (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    fov_h_scale = ttk.Scale(parent, from_=60, to=360, variable=self.params['fov_horizontal'], orient=tk.HORIZONTAL, length=200)
    fov_h_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    fov_h_scale.bind('<ButtonRelease-1>', self.on_param_change)
    row += 1
    
    fov_h_control_frame = ttk.Frame(parent)
    fov_h_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(fov_h_control_frame, text="-", width=3, command=lambda: self.adjust_fov('horizontal', -30)).pack(side=tk.LEFT, padx=(0, 2))
    fov_h_entry = ttk.Entry(fov_h_control_frame, textvariable=self.params['fov_horizontal'], width=8)
    fov_h_entry.pack(side=tk.LEFT, padx=2)
    fov_h_entry.bind('<Return>', self.on_param_change)
    ttk.Button(fov_h_control_frame, text="+", width=3, command=lambda: self.adjust_fov('horizontal', 30)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Vertical FOV
    ttk.Label(parent, text="Vertical FOV (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    fov_v_scale = ttk.Scale(parent, from_=30, to=180, variable=self.params['fov_vertical'], orient=tk.HORIZONTAL, length=200)
    fov_v_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    fov_v_scale.bind('<ButtonRelease-1>', self.on_param_change)
    row += 1
    
    fov_v_control_frame = ttk.Frame(parent)
    fov_v_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(fov_v_control_frame, text="-", width=3, command=lambda: self.adjust_fov('vertical', -15)).pack(side=tk.LEFT, padx=(0, 2))
    fov_v_entry = ttk.Entry(fov_v_control_frame, textvariable=self.params['fov_vertical'], width=8)
    fov_v_entry.pack(side=tk.LEFT, padx=2)
    fov_v_entry.bind('<Return>', self.on_param_change)
    ttk.Button(fov_v_control_frame, text="+", width=3, command=lambda: self.adjust_fov('vertical', 15)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Projection options
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Projection Options", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    # Allow behind camera checkbox
    behind_camera_check = ttk.Checkbutton(parent, text="Allow behind camera content", 
                                        variable=self.params['allow_behind_camera'], 
                                        command=self.on_param_change)
    behind_camera_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0), pady=2)
    row += 1
    
    # Action buttons
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    button_frame = ttk.Frame(parent)
    button_frame.grid(row=row, column=0, columnspan=2, pady=10)
    
    ttk.Button(button_frame, text="Reset", command=self.reset_params).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
    
    # Presets
    row += 1
    ttk.Label(parent, text="Presets", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
    row += 1
    
    preset_frame = ttk.Frame(parent)
    preset_frame.grid(row=row, column=0, columnspan=2, pady=5)
    
    ttk.Button(preset_frame, text="Full Sphere", command=lambda: self.apply_preset('full_sphere')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Hemisphere", command=lambda: self.apply_preset('hemisphere')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Panoramic", command=lambda: self.apply_preset('panoramic')).pack(side=tk.LEFT, padx=2)
    
    row += 1
    preset_frame2 = ttk.Frame(parent)
    preset_frame2.grid(row=row, column=0, columnspan=2, pady=5)
    
    ttk.Button(preset_frame2, text="Equatorial", command=lambda: self.apply_preset('equatorial')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Look Up", command=lambda: self.apply_preset('look_up')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Look Down", command=lambda: self.apply_preset('look_down')).pack(side=tk.LEFT, padx=2)
    
  def setup_image_display(self, parent: ttk.Widget) -> None:
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
    
    self.original_label = ttk.Label(left_frame)
    self.original_label.pack()
    
    # Right side - Projected image
    right_frame = ttk.Frame(display_frame)
    right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
    
    ttk.Label(right_frame, text="Spherical Projection Result", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
    
    self.projected_label = ttk.Label(right_frame)
    self.projected_label.pack()
    
    # Terminal-like output area at bottom spanning both columns
    terminal_frame = ttk.LabelFrame(display_frame, text="Program Output", padding="5")
    terminal_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
    terminal_frame.columnconfigure(0, weight=1)
    terminal_frame.rowconfigure(0, weight=1)
    
    # Create terminal-like text widget with scrollbar
    self.terminal_text = tk.Text(
      terminal_frame,
      bg='black',
      fg='#00ff00',  # Green text like classic terminals
      font=('Consolas', 9),
      wrap=tk.WORD,
      state=tk.DISABLED,  # Read-only
      cursor='arrow'
    )
    
    terminal_scrollbar = ttk.Scrollbar(terminal_frame, orient=tk.VERTICAL, command=self.terminal_text.yview)
    self.terminal_text.configure(yscrollcommand=terminal_scrollbar.set)
    
    self.terminal_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    terminal_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    # Initialize with welcome message
    self.log_message("Fisheye Spherical Projection Tool initialized")
    self.log_message("Ready for processing...")
  
  def on_param_change(self, event: Optional[tk.Event] = None) -> None:
    # Debounce rapid changes - delay only after last input
    # Cancel any existing scheduled update
    if hasattr(self, 'pending_update_id') and self.pending_update_id:
      self.root.after_cancel(self.pending_update_id)
    
    # Schedule new update after 100ms delay from this input
    self.pending_update_id = self.root.after(100, self.delayed_update)
  
  def delayed_update(self) -> None:
    self.pending_update_id = None
    self.update_images()
  
  def update_images(self) -> None:
    # Update original image display
    self.update_original_image()
    
    # Start background processing for spherical projection
    if self.processing_thread and self.processing_thread.is_alive():
      # Previous processing still running, queue this update
      try:
        self.processing_queue.put_nowait('update')
      except queue.Full:
        pass  # Queue is full, skip this update
    else:
      # Start new processing thread
      self.processing_thread = threading.Thread(target=self.process_spherical_projection)
      self.processing_thread.daemon = True
      self.processing_thread.start()
  
  def update_original_image(self) -> None:
    # Calculate optimal size for side-by-side display
    # Assume window width of 1600, controls take 350, leaving ~1250 for images
    # Each image gets ~600 pixels width with padding
    display_size = 600
    
    # Resize and display original fisheye image
    img_resized = cv2.resize(self.fisheye_img, (display_size, display_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    self.original_label.configure(image=img_tk)
    self.original_label.image = img_tk  # Keep a reference
  
  def log_message(self, message: str) -> None:
    """Add a message to the terminal output with timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    
    # Enable editing to add text
    self.terminal_text.configure(state=tk.NORMAL)
    self.terminal_text.insert(tk.END, formatted_message)
    
    # Auto-scroll to bottom
    self.terminal_text.see(tk.END)
    
    # Disable editing again
    self.terminal_text.configure(state=tk.DISABLED)
    
    # Update the display
    self.terminal_text.update_idletasks()
  
  def process_spherical_projection(self) -> None:
    try:
      self.log_message("Starting spherical projection processing...")
      
      # Get current parameter values
      params = {k: v.get() for k, v in self.params.items()}
      
      # Log current parameters
      self.log_message(f"Parameters: Yaw={params['yaw_offset']:.1f}°, Pitch={params['pitch_offset']:.1f}°")
      self.log_message(f"FOV: H={params['fov_horizontal']:.1f}°, V={params['fov_vertical']:.1f}°")
      
      # Generate spherical projection using the SphericalProjection class
      projected_img = self.projector.project(
        self.fisheye_img,
        output_width=params['output_width'],
        output_height=params['output_height'],
        yaw_offset=params['yaw_offset'],
        pitch_offset=params['pitch_offset'],
        fov_horizontal=params['fov_horizontal'],
        fov_vertical=params['fov_vertical'],
        allow_behind_camera=params['allow_behind_camera']
      )
      
      # Check if there's a newer update request
      try:
        self.processing_queue.get_nowait()
        # There's a newer request, restart processing
        self.log_message("New parameter change detected, restarting processing...")
        self.process_spherical_projection()
        return
      except queue.Empty:
        pass
      
      # Put result in queue for main thread
      self.result_queue.put(projected_img)
      
    except Exception as e:
      error_msg = f"Processing error: {e}"
      self.log_message(error_msg)
      self.result_queue.put(f"Error: {e}")
  
  def check_results(self) -> None:
    try:
      result = self.result_queue.get_nowait()
      if isinstance(result, str) and result.startswith("Error"):
        self.log_message(result)
      else:
        self.update_projected_image(result)
        self.log_message("Spherical projection completed successfully")
    except queue.Empty:
      pass
    
    # Schedule next check
    self.root.after(100, self.check_results)
  
  def update_projected_image(self, projected_img: np.ndarray) -> None:
    # Calculate optimal size for side-by-side display
    # For spherical projections, we need to handle panoramic aspect ratios
    display_width = 600
    h, w = projected_img.shape[:2]
    
    # Calculate aspect ratio and resize accordingly
    aspect_ratio = w / h
    if aspect_ratio > 1:  # Wide panoramic image
      new_w = display_width
      new_h = int(display_width / aspect_ratio)
    else:  # Tall or square image
      new_h = display_width
      new_w = int(display_width * aspect_ratio)
    
    # Ensure minimum size for visibility
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
    
    self.projected_label.configure(image=img_tk)
    self.projected_label.image = img_tk  # Keep a reference
    
    # Store the full-size result for saving
    self.last_projected_img = projected_img
  
  def reset_params(self) -> None:
    self.log_message("Resetting all parameters to default values")
    self.params['output_width'].set(2048)
    self.params['output_height'].set(1024)
    self.params['yaw_offset'].set(0.0)
    self.params['pitch_offset'].set(0.0)
    self.params['fov_horizontal'].set(360.0)
    self.params['fov_vertical'].set(180.0)
    self.params['allow_behind_camera'].set(True)
    self.update_images()
  
  def apply_preset(self, preset_type: str) -> None:
    self.log_message(f"Applying '{preset_type}' preset")
    if preset_type == 'full_sphere':
      self.params['fov_horizontal'].set(360.0)
      self.params['fov_vertical'].set(180.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(0.0)
      self.params['output_width'].set(2048)
      self.params['output_height'].set(1024)
    elif preset_type == 'hemisphere':
      self.params['fov_horizontal'].set(360.0)
      self.params['fov_vertical'].set(90.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(0.0)
      self.params['output_width'].set(2048)
      self.params['output_height'].set(512)
    elif preset_type == 'panoramic':
      self.params['fov_horizontal'].set(360.0)
      self.params['fov_vertical'].set(120.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(0.0)
      self.params['output_width'].set(3600)
      self.params['output_height'].set(1200)
    elif preset_type == 'equatorial':
      self.params['fov_horizontal'].set(360.0)
      self.params['fov_vertical'].set(60.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(0.0)
      self.params['output_width'].set(3600)
      self.params['output_height'].set(600)
    elif preset_type == 'look_up':
      self.params['fov_horizontal'].set(180.0)
      self.params['fov_vertical'].set(90.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(45.0)
      self.params['output_width'].set(1800)
      self.params['output_height'].set(900)
    elif preset_type == 'look_down':
      self.params['fov_horizontal'].set(180.0)
      self.params['fov_vertical'].set(90.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(-45.0)
      self.params['output_width'].set(1800)
      self.params['output_height'].set(900)
    
    self.update_images()
  
  def adjust_rotation(self, axis: str, delta: float) -> None:
    """Adjust rotation value by delta degrees with proper range clamping."""
    if axis == 'yaw':
      current = self.params['yaw_offset'].get()
      new_value = current + delta
      # Clamp to -180 to 180 range
      if new_value > 180:
        new_value -= 360
      elif new_value < -180:
        new_value += 360
      self.params['yaw_offset'].set(new_value)
    elif axis == 'pitch':
      current = self.params['pitch_offset'].get()
      new_value = max(-90, min(90, current + delta))  # Clamp to -90 to 90 range
      self.params['pitch_offset'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_param_change()
  
  def adjust_fov(self, axis: str, delta: float) -> None:
    """Adjust field of view value by delta degrees with proper range clamping."""
    if axis == 'horizontal':
      current = self.params['fov_horizontal'].get()
      new_value = max(60, min(360, current + delta))  # Clamp to 60 to 360 range
      self.params['fov_horizontal'].set(new_value)
    elif axis == 'vertical':
      current = self.params['fov_vertical'].get()
      new_value = max(30, min(180, current + delta))  # Clamp to 30 to 180 range
      self.params['fov_vertical'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_param_change()
  
  def save_image(self) -> None:
    if hasattr(self, 'last_projected_img'):
      filename = f"spherical_yaw{self.params['yaw_offset'].get():.0f}_pitch{self.params['pitch_offset'].get():.0f}_fovh{self.params['fov_horizontal'].get():.0f}_fovv{self.params['fov_vertical'].get():.0f}.jpg"
      cv2.imwrite(filename, self.last_projected_img)
      self.log_message(f"Image saved successfully: {filename}")
      messagebox.showinfo("Success", f"Image saved as {filename}")
    else:
      self.log_message("Save failed: No projected image available")
      messagebox.showwarning("Warning", "No projected image to save")

def main() -> None:
  root = tk.Tk()
  app = SphericalFisheyeUI(root)
  root.mainloop()

if __name__ == "__main__":
  main()
