import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from fisheye_rectify import parse_camera_params, perspective_projection
import threading
import queue

class FisheyeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fisheye Perspective Projection Tool")
        self.root.geometry("1600x900")
        
        # Load camera parameters and fisheye image
        try:
            self.camera_params = parse_camera_params("cameras.txt")
            self.fisheye_img = cv2.imread("fisheye_img.jpg")
            if self.fisheye_img is None:
                raise FileNotFoundError("Could not load fisheye_img.jpg")
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
            'virtual_fy': tk.DoubleVar(value=0.0)   # 0 means auto-calculate
        }
        
        # Threading for image processing
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.update_pending = False
        
        self.setup_ui()
        self.update_images()
        
        # Start periodic check for results
        self.root.after(100, self.check_results)
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Perspective Projection Parameters", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for images
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Setup control widgets
        self.setup_controls(control_frame)
        
        # Setup image display
        self.setup_image_display(image_frame)
    
    def setup_controls(self, parent):
        row = 0
        
        # Output dimensions
        ttk.Label(parent, text="Output Dimensions", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1
        
        ttk.Label(parent, text="Width:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        width_spinbox = ttk.Spinbox(parent, from_=200, to=2000, increment=50, textvariable=self.params['output_width'], width=10)
        width_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
        width_spinbox.bind('<KeyRelease>', self.on_param_change)
        width_spinbox.bind('<<Increment>>', self.on_param_change)
        width_spinbox.bind('<<Decrement>>', self.on_param_change)
        row += 1
        
        ttk.Label(parent, text="Height:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        height_spinbox = ttk.Spinbox(parent, from_=200, to=2000, increment=50, textvariable=self.params['output_height'], width=10)
        height_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2)
        height_spinbox.bind('<KeyRelease>', self.on_param_change)
        height_spinbox.bind('<<Increment>>', self.on_param_change)
        height_spinbox.bind('<<Decrement>>', self.on_param_change)
        row += 1
        
        # Rotation controls
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        ttk.Label(parent, text="Rotation Controls", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1
        
        # Yaw
        ttk.Label(parent, text="Yaw (°):").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        yaw_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.params['yaw_offset'], orient=tk.HORIZONTAL, length=200)
        yaw_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        yaw_scale.bind('<Motion>', self.on_param_change)
        yaw_scale.bind('<ButtonRelease-1>', self.on_param_change)
        row += 1
        
        yaw_entry = ttk.Entry(parent, textvariable=self.params['yaw_offset'], width=10)
        yaw_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        yaw_entry.bind('<KeyRelease>', self.on_param_change)
        row += 1
        
        # Pitch
        ttk.Label(parent, text="Pitch (°):").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        pitch_scale = ttk.Scale(parent, from_=-90, to=90, variable=self.params['pitch_offset'], orient=tk.HORIZONTAL, length=200)
        pitch_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        pitch_scale.bind('<Motion>', self.on_param_change)
        pitch_scale.bind('<ButtonRelease-1>', self.on_param_change)
        row += 1
        
        pitch_entry = ttk.Entry(parent, textvariable=self.params['pitch_offset'], width=10)
        pitch_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        pitch_entry.bind('<KeyRelease>', self.on_param_change)
        row += 1
        
        # Roll
        ttk.Label(parent, text="Roll (°):").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        roll_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.params['roll_offset'], orient=tk.HORIZONTAL, length=200)
        roll_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        roll_scale.bind('<Motion>', self.on_param_change)
        roll_scale.bind('<ButtonRelease-1>', self.on_param_change)
        row += 1
        
        roll_entry = ttk.Entry(parent, textvariable=self.params['roll_offset'], width=10)
        roll_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        roll_entry.bind('<KeyRelease>', self.on_param_change)
        row += 1
        
        # Field of view
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        ttk.Label(parent, text="Field of View", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1
        
        ttk.Label(parent, text="Horizontal FOV (°):").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        fov_scale = ttk.Scale(parent, from_=10, to=150, variable=self.params['fov_horizontal'], orient=tk.HORIZONTAL, length=200)
        fov_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        fov_scale.bind('<Motion>', self.on_param_change)
        fov_scale.bind('<ButtonRelease-1>', self.on_param_change)
        row += 1
        
        fov_entry = ttk.Entry(parent, textvariable=self.params['fov_horizontal'], width=10)
        fov_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        fov_entry.bind('<KeyRelease>', self.on_param_change)
        row += 1
        
        # Virtual camera parameters
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        ttk.Label(parent, text="Virtual Camera (0 = auto)", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1
        
        ttk.Label(parent, text="Virtual fx:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        fx_entry = ttk.Entry(parent, textvariable=self.params['virtual_fx'], width=10)
        fx_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        fx_entry.bind('<KeyRelease>', self.on_param_change)
        row += 1
        
        ttk.Label(parent, text="Virtual fy:").grid(row=row, column=0, sticky=tk.W, padx=(10, 5))
        fy_entry = ttk.Entry(parent, textvariable=self.params['virtual_fy'], width=10)
        fy_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        fy_entry.bind('<KeyRelease>', self.on_param_change)
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
    
    def setup_image_display(self, parent):
        # Image display frame
        display_frame = ttk.Frame(parent)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for side-by-side layout
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
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
        
        # Status label at bottom spanning both columns
        self.status_label = ttk.Label(display_frame, text="Ready", font=('Arial', 10))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(10, 0))
    
    def on_param_change(self, event=None):
        # Debounce rapid changes
        if not self.update_pending:
            self.update_pending = True
            self.root.after(100, self.delayed_update)  # 100ms delay
    
    def delayed_update(self):
        self.update_pending = False
        self.update_images()
    
    def update_images(self):
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
    
    def update_original_image(self):
        # Resize and display original fisheye image
        img_resized = cv2.resize(self.fisheye_img, (800, 800))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.original_label.configure(image=img_tk)
        self.original_label.image = img_tk  # Keep a reference
    
    def process_perspective_projection(self):
        try:
            self.status_label.configure(text="Processing...")
            
            # Get current parameter values
            params = {k: v.get() for k, v in self.params.items()}
            
            # Handle auto-calculation of virtual focal lengths
            virtual_fx = params['virtual_fx'] if params['virtual_fx'] > 0 else None
            virtual_fy = params['virtual_fy'] if params['virtual_fy'] > 0 else None
            
            # Generate perspective projection
            projected_img = perspective_projection(
                "fisheye_img.jpg",
                self.camera_params,
                output_width=params['output_width'],
                output_height=params['output_height'],
                yaw_offset=params['yaw_offset'],
                pitch_offset=params['pitch_offset'],
                roll_offset=params['roll_offset'],
                fov_horizontal=params['fov_horizontal'],
                virtual_fx=virtual_fx,
                virtual_fy=virtual_fy
            )
            
            # Check if there's a newer update request
            try:
                self.processing_queue.get_nowait()
                # There's a newer request, restart processing
                self.process_perspective_projection()
                return
            except queue.Empty:
                pass
            
            # Put result in queue for main thread
            self.result_queue.put(projected_img)
            
        except Exception as e:
            self.result_queue.put(f"Error: {e}")
    
    def check_results(self):
        try:
            result = self.result_queue.get_nowait()
            if isinstance(result, str) and result.startswith("Error"):
                self.status_label.configure(text=result)
            else:
                self.update_projected_image(result)
                self.status_label.configure(text="Ready")
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_results)
    
    def update_projected_image(self, projected_img):
        # Resize for display (max 800x800 while maintaining aspect ratio)
        h, w = projected_img.shape[:2]
        max_size = 800
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        
        img_resized = cv2.resize(projected_img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.projected_label.configure(image=img_tk)
        self.projected_label.image = img_tk  # Keep a reference
        
        # Store the full-size result for saving
        self.last_projected_img = projected_img
    
    def reset_params(self):
        self.params['output_width'].set(800)
        self.params['output_height'].set(800)
        self.params['yaw_offset'].set(0.0)
        self.params['pitch_offset'].set(0.0)
        self.params['roll_offset'].set(0.0)
        self.params['fov_horizontal'].set(90.0)
        self.params['virtual_fx'].set(0.0)
        self.params['virtual_fy'].set(0.0)
        self.update_images()
    
    def apply_preset(self, preset_type):
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
    
    def save_image(self):
        if hasattr(self, 'last_projected_img'):
            filename = f"projection_yaw{self.params['yaw_offset'].get():.0f}_pitch{self.params['pitch_offset'].get():.0f}_roll{self.params['roll_offset'].get():.0f}_fov{self.params['fov_horizontal'].get():.0f}.jpg"
            cv2.imwrite(filename, self.last_projected_img)
            messagebox.showinfo("Success", f"Image saved as {filename}")
        else:
            messagebox.showwarning("Warning", "No projected image to save")

def main():
    root = tk.Tk()
    app = FisheyeUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
