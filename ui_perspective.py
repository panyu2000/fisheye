import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from camera_params import parse_camera_params
from perspective_projection import PerspectiveProjection
import threading
import queue
from typing import Dict, Union, Optional

class FisheyeUI:
  def __init__(self, root: tk.Tk) -> None:
    self.root = root
    self.root.title("Fisheye Perspective Projection Tool")
    self.root.geometry("1600x1000")
    
    # Load camera parameters and fisheye image
    try:
      self.camera_params = parse_camera_params("camera_intrinsics.yaml")
      self.fisheye_img = cv2.imread("fisheye_img.jpg")
      if self.fisheye_img is None:
        raise FileNotFoundError("Could not load fisheye_img.jpg")
        
      # Get image dimensions for validation
      img_height, img_width = self.fisheye_img.shape[:2]
      
      # Create PerspectiveProjection instance with caching capabilities
      # Use vectorized=True by default for fast performance
      self.projector = PerspectiveProjection(self.camera_params, input_image_size=(img_width, img_height), use_vectorized=True)
      
    except Exception as e:
      messagebox.showerror("Error", f"Failed to load files: {e}")
      return
    
    # Variables for perspective projection parameters
    self.params = {
      'output_width': tk.IntVar(value=800),
      'output_height': tk.IntVar(value=800),
      'yaw_offset': tk.DoubleVar(value=0.0),
      'pitch_offset': tk.DoubleVar(value=0.0),
      'roll_offset': tk.DoubleVar(value=0.0),
      'fov_horizontal': tk.DoubleVar(value=90.0),
      'virtual_fx': tk.DoubleVar(value=0.0),  # 0 means auto-calculate
      'virtual_fy': tk.DoubleVar(value=0.0),  # 0 means auto-calculate
      'allow_behind_camera': tk.BooleanVar(value=True)  # True enables behind camera content
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
    control_frame = ttk.LabelFrame(main_frame, text="Perspective Projection Parameters", padding="10")
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
    width_spinbox = ttk.Spinbox(parent, from_=200, to=2000, increment=50, textvariable=self.params['output_width'], width=10)
    width_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
    width_spinbox.bind('<Return>', self.on_param_change)
    width_spinbox.bind('<<Increment>>', self.on_param_change)
    width_spinbox.bind('<<Decrement>>', self.on_param_change)
    row += 1
    
    ttk.Label(parent, text="Height:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    height_spinbox = ttk.Spinbox(parent, from_=200, to=2000, increment=50, textvariable=self.params['output_height'], width=10)
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
    
    # Roll
    ttk.Label(parent, text="Roll (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    roll_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.params['roll_offset'], orient=tk.HORIZONTAL, length=200)
    roll_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    roll_scale.bind('<ButtonRelease-1>', self.on_param_change)
    row += 1
    
    roll_control_frame = ttk.Frame(parent)
    roll_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(roll_control_frame, text="-", width=3, command=lambda: self.adjust_rotation('roll', -10)).pack(side=tk.LEFT, padx=(0, 2))
    roll_entry = ttk.Entry(roll_control_frame, textvariable=self.params['roll_offset'], width=8)
    roll_entry.pack(side=tk.LEFT, padx=2)
    roll_entry.bind('<Return>', self.on_param_change)
    ttk.Button(roll_control_frame, text="+", width=3, command=lambda: self.adjust_rotation('roll', 10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Field of view
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Field of View", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    # Horizontal FOV
    ttk.Label(parent, text="Horizontal FOV (°):").grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
    row += 1
    fov_scale = ttk.Scale(parent, from_=10, to=175, variable=self.params['fov_horizontal'], orient=tk.HORIZONTAL, length=200)
    fov_scale.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
    fov_scale.bind('<ButtonRelease-1>', self.on_param_change)
    row += 1
    
    fov_control_frame = ttk.Frame(parent)
    fov_control_frame.grid(row=row, column=0, columnspan=2, pady=2)
    ttk.Button(fov_control_frame, text="-", width=3, command=lambda: self.adjust_fov(-10)).pack(side=tk.LEFT, padx=(0, 2))
    fov_entry = ttk.Entry(fov_control_frame, textvariable=self.params['fov_horizontal'], width=8)
    fov_entry.pack(side=tk.LEFT, padx=2)
    fov_entry.bind('<Return>', self.on_param_change)
    ttk.Button(fov_control_frame, text="+", width=3, command=lambda: self.adjust_fov(10)).pack(side=tk.LEFT, padx=(2, 0))
    row += 1
    
    # Virtual camera parameters
    ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
    row += 1
    
    ttk.Label(parent, text="Virtual Camera (0 = auto)", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    row += 1
    
    ttk.Label(parent, text="Virtual fx:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    fx_entry = ttk.Entry(parent, textvariable=self.params['virtual_fx'], width=10)
    fx_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
    fx_entry.bind('<Return>', self.on_param_change)
    row += 1
    
    ttk.Label(parent, text="Virtual fy:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
    fy_entry = ttk.Entry(parent, textvariable=self.params['virtual_fy'], width=10)
    fy_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
    fy_entry.bind('<Return>', self.on_param_change)
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
    
    ttk.Button(preset_frame, text="Wide Angle", command=lambda: self.apply_preset('wide')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Standard", command=lambda: self.apply_preset('standard')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame, text="Telephoto", command=lambda: self.apply_preset('telephoto')).pack(side=tk.LEFT, padx=2)
    
    row += 1
    preset_frame2 = ttk.Frame(parent)
    preset_frame2.grid(row=row, column=0, columnspan=2, pady=5)
    
    ttk.Button(preset_frame2, text="Look Up", command=lambda: self.apply_preset('up')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Look Down", command=lambda: self.apply_preset('down')).pack(side=tk.LEFT, padx=2)
    ttk.Button(preset_frame2, text="Side View", command=lambda: self.apply_preset('side')).pack(side=tk.LEFT, padx=2)
    
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
    
    ttk.Label(right_frame, text="Perspective Projection Result", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
    
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
    self.log_message("Fisheye Perspective Projection Tool initialized")
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
    
    # Start background processing for perspective projection
    if self.processing_thread and self.processing_thread.is_alive():
      # Previous processing still running, queue this update
      try:
        self.processing_queue.put_nowait('update')
      except queue.Full:
        pass  # Queue is full, skip this update
    else:
      # Start new processing thread
      self.processing_thread = threading.Thread(target=self.process_perspective_projection)
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
  
  def process_perspective_projection(self) -> None:
    try:
      self.log_message("Starting projection processing...")
      
      # Get current parameter values
      params = {k: v.get() for k, v in self.params.items()}
      
      # Log current parameters
      self.log_message(f"Parameters: Yaw={params['yaw_offset']:.1f}°, Pitch={params['pitch_offset']:.1f}°, Roll={params['roll_offset']:.1f}°, FOV={params['fov_horizontal']:.1f}°")
      
      # Handle auto-calculation of virtual focal lengths
      virtual_fx = params['virtual_fx'] if params['virtual_fx'] > 0 else None
      virtual_fy = params['virtual_fy'] if params['virtual_fy'] > 0 else None
      
      # Generate perspective projection using the PerspectiveProjection class
      projected_img = self.projector.project(
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
      
      # Check if there's a newer update request
      try:
        self.processing_queue.get_nowait()
        # There's a newer request, restart processing
        self.log_message("New parameter change detected, restarting processing...")
        self.process_perspective_projection()
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
        self.log_message("Projection completed successfully")
    except queue.Empty:
      pass
    
    # Schedule next check
    self.root.after(100, self.check_results)
  
  def update_projected_image(self, projected_img: np.ndarray) -> None:
    # Calculate optimal size for side-by-side display to match original image
    display_size = 600
    h, w = projected_img.shape[:2]
    
    # Maintain aspect ratio while fitting within display_size
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
    
    self.projected_label.configure(image=img_tk)
    self.projected_label.image = img_tk  # Keep a reference
    
    # Store the full-size result for saving
    self.last_projected_img = projected_img
  
  def reset_params(self) -> None:
    self.log_message("Resetting all parameters to default values")
    self.params['output_width'].set(800)
    self.params['output_height'].set(800)
    self.params['yaw_offset'].set(0.0)
    self.params['pitch_offset'].set(0.0)
    self.params['roll_offset'].set(0.0)
    self.params['fov_horizontal'].set(90.0)
    self.params['virtual_fx'].set(0.0)
    self.params['virtual_fy'].set(0.0)
    self.params['allow_behind_camera'].set(True)
    self.update_images()
  
  def apply_preset(self, preset_type: str) -> None:
    self.log_message(f"Applying '{preset_type}' preset")
    if preset_type == 'wide':
      self.params['fov_horizontal'].set(120.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(0.0)
      self.params['roll_offset'].set(0.0)
    elif preset_type == 'standard':
      self.params['fov_horizontal'].set(60.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(0.0)
      self.params['roll_offset'].set(0.0)
    elif preset_type == 'telephoto':
      self.params['fov_horizontal'].set(30.0)
      self.params['yaw_offset'].set(0.0)
      self.params['pitch_offset'].set(0.0)
      self.params['roll_offset'].set(0.0)
    elif preset_type == 'up':
      self.params['pitch_offset'].set(45.0)
      self.params['yaw_offset'].set(0.0)
      self.params['roll_offset'].set(0.0)
    elif preset_type == 'down':
      self.params['pitch_offset'].set(-45.0)
      self.params['yaw_offset'].set(0.0)
      self.params['roll_offset'].set(0.0)
    elif preset_type == 'side':
      self.params['yaw_offset'].set(90.0)
      self.params['pitch_offset'].set(0.0)
      self.params['roll_offset'].set(0.0)
    
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
    elif axis == 'roll':
      current = self.params['roll_offset'].get()
      new_value = current + delta
      # Clamp to -180 to 180 range
      if new_value > 180:
        new_value -= 360
      elif new_value < -180:
        new_value += 360
      self.params['roll_offset'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_param_change()
  
  def adjust_fov(self, delta: float) -> None:
    """Adjust field of view value by delta degrees with proper range clamping."""
    current = self.params['fov_horizontal'].get()
    new_value = max(10, min(175, current + delta))  # Clamp to 10 to 175 range
    self.params['fov_horizontal'].set(new_value)
    
    # Trigger debounced update for button controls
    self.on_param_change()
  
  def save_image(self) -> None:
    if hasattr(self, 'last_projected_img'):
      filename = f"projection_yaw{self.params['yaw_offset'].get():.0f}_pitch{self.params['pitch_offset'].get():.0f}_roll{self.params['roll_offset'].get():.0f}_fov{self.params['fov_horizontal'].get():.0f}.jpg"
      cv2.imwrite(filename, self.last_projected_img)
      self.log_message(f"Image saved successfully: {filename}")
      messagebox.showinfo("Success", f"Image saved as {filename}")
    else:
      self.log_message("Save failed: No projected image available")
      messagebox.showwarning("Warning", "No projected image to save")

def main() -> None:
  root = tk.Tk()
  app = FisheyeUI(root)
  root.mainloop()

if __name__ == "__main__":
  main()
