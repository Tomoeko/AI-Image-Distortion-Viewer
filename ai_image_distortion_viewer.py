import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageEnhance, ImageChops
from tkinterdnd2 import DND_FILES, TkinterDnD
import platform
import io
from abc import ABC, abstractmethod

try:
    import numpy as np
    from scipy.fft import fft2, fftshift
except ImportError:
    print("Error: NumPy and SciPy are required for advanced features.")
    print("Please install them using: pip install numpy scipy")
    exit()


# ==================== Filter Base Class ====================
class DistortionFilter(ABC):
    """Abstract base class for all distortion filters"""
    
    @abstractmethod
    def get_name(self):
        """Return the display name of the filter"""
        pass
    
    @abstractmethod
    def apply(self, image):
        """Apply the filter to the image and return the result"""
        pass


# ==================== Concrete Filter Implementations ====================
class EdgeDetectionFilter(DistortionFilter):
    def get_name(self):
        return "Edge Detection"
    
    def apply(self, image):
        distorted = ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES)
        return ImageOps.autocontrast(distorted, cutoff=5).convert("RGB")


class ColorEmbossFilter(DistortionFilter):
    def get_name(self):
        return "Color Emboss"
    
    def apply(self, image):
        return image.filter(ImageFilter.EMBOSS)


class SolarizeFilter(DistortionFilter):
    def get_name(self):
        return "Solarize"
    
    def apply(self, image):
        return ImageOps.solarize(image, threshold=128)


class FrequencyFFTFilter(DistortionFilter):
    def get_name(self):
        return "Frequency (FFT)"
    
    def apply(self, image):
        gray_image = image.convert("L")
        np_image = np.array(gray_image)
        f_transform = fft2(np_image)
        f_transform_shifted = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
        magnitude_spectrum = 255 * (magnitude_spectrum / np.max(magnitude_spectrum))
        return Image.fromarray(magnitude_spectrum.astype(np.uint8)).convert("RGB")


class NoiseResidualFilter(DistortionFilter):
    def get_name(self):
        return "Noise Residual"
    
    def apply(self, image):
        denoised = image.filter(ImageFilter.GaussianBlur(radius=1))
        return ImageChops.subtract(image, denoised, scale=2.0, offset=128).convert("RGB")


class ELAFilter(DistortionFilter):
    def get_name(self):
        return "ELA (Error Level Analysis)"
    
    def apply(self, image):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=95)
        buffer.seek(0)
        resaved_image = Image.open(buffer)
        diff = ImageChops.difference(image, resaved_image)
        return ImageOps.autocontrast(diff, cutoff=2).convert("RGB")


class ELALuminanceFilter(DistortionFilter):
    def get_name(self):
        return "ELA 2 (Luminance)"
    
    def apply(self, image):
        original_lum = image.convert("L")
        buffer = io.BytesIO()
        original_lum.save(buffer, "JPEG", quality=95)
        buffer.seek(0)
        resaved_lum = Image.open(buffer)
        diff = ImageChops.difference(original_lum, resaved_lum)
        return ImageOps.autocontrast(diff, cutoff=2).convert("RGB")


class ColorDiscrepancyFilter(DistortionFilter):
    def get_name(self):
        return "Color Discrepancy"
    
    def apply(self, image):
        np_image = np.array(image, dtype=np.float32)
        std_map = np.std(np_image, axis=2)
        std_map = 255 * (std_map / np.max(std_map) if np.max(std_map) > 0 else 0)
        return Image.fromarray(std_map.astype(np.uint8)).convert("RGB")


# ==================== Filter Registry ====================
class FilterRegistry:
    """Manages all available filters"""
    
    def __init__(self):
        self.filters = [
            EdgeDetectionFilter(),
            ColorEmbossFilter(),
            SolarizeFilter(),
            FrequencyFFTFilter(),
            NoiseResidualFilter(),
            ELAFilter(),
            ELALuminanceFilter(),
            ColorDiscrepancyFilter()
        ]
        self._filter_map = {f.get_name(): f for f in self.filters}
    
    def get_filter_names(self):
        return [f.get_name() for f in self.filters]
    
    def get_filter(self, name):
        return self._filter_map.get(name)


# ==================== Image State Manager ====================
class ImageState:
    """Manages the state of loaded images and transformations"""
    
    def __init__(self):
        self.original_image = None
        self.base_distorted_image = None
        self.enhanced_distorted_image = None
        self.image_path = None
    
    def load_image(self, path):
        self.image_path = path
        self.original_image = Image.open(path).convert("RGB")
        self.base_distorted_image = None
        self.enhanced_distorted_image = None
    
    def has_image(self):
        return self.original_image is not None
    
    def clear(self):
        self.original_image = None
        self.base_distorted_image = None
        self.enhanced_distorted_image = None
        self.image_path = None


# ==================== View Controller ====================
class ViewController:
    """Manages zoom, pan, and view state"""
    
    def __init__(self, min_zoom=1.0, max_zoom=10.0):
        self.zoom_level = min_zoom
        self.max_zoom = max_zoom
        self.min_zoom = min_zoom
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.is_panning = False
        self.last_drag_x = 0
        self.last_drag_y = 0
    
    def reset_for_image(self, image, canvas_width, canvas_height):
        """Calculate and set the initial zoom level for an image"""
        ratio = min(canvas_width / image.width, canvas_height / image.height)
        self.min_zoom = ratio
        self.zoom_level = self.min_zoom
        self.view_offset_x = 0
        self.view_offset_y = 0
    
    def set_zoom(self, new_zoom, anchor_x, anchor_y):
        """Set zoom level with anchor point"""
        old_zoom = self.zoom_level
        self.zoom_level = max(self.min_zoom, min(new_zoom, self.max_zoom))
        self.view_offset_x = (self.view_offset_x + anchor_x) * (self.zoom_level / old_zoom) - anchor_x
        self.view_offset_y = (self.view_offset_y + anchor_y) * (self.zoom_level / old_zoom) - anchor_y
        return self.zoom_level
    
    def update_min_zoom(self, new_min_zoom):
        """Update minimum zoom and adjust current zoom if needed"""
        if self.zoom_level <= self.min_zoom:
            self.zoom_level = new_min_zoom
        self.min_zoom = new_min_zoom
    
    def is_zoomed(self):
        return self.zoom_level > self.min_zoom
    
    def start_pan(self, x, y):
        self.is_panning = True
        self.last_drag_x = x
        self.last_drag_y = y
    
    def pan(self, x, y):
        if self.is_panning:
            dx = x - self.last_drag_x
            dy = y - self.last_drag_y
            self.view_offset_x -= dx
            self.view_offset_y -= dy
            self.last_drag_x = x
            self.last_drag_y = y
            return True
        return False
    
    def end_pan(self):
        self.is_panning = False
    
    def constrain_offsets(self, zoomed_width, zoomed_height, canvas_width, canvas_height):
        """Constrain view offsets to valid ranges"""
        self.view_offset_x = max(0, min(self.view_offset_x, zoomed_width - canvas_width))
        self.view_offset_y = max(0, min(self.view_offset_y, zoomed_height - canvas_height))


# ==================== Image Processor ====================
class ImageProcessor:
    """Handles image processing and enhancement operations"""
    
    def __init__(self, filter_registry):
        self.filter_registry = filter_registry
        self.brightness = 1.0
        self.contrast = 1.0
        self.sharpness = 1.0
    
    def apply_filter(self, image, filter_name):
        """Apply a distortion filter to an image"""
        filter_obj = self.filter_registry.get_filter(filter_name)
        if filter_obj:
            return filter_obj.apply(image)
        return image
    
    def apply_enhancements(self, image):
        """Apply brightness, contrast, and sharpness enhancements"""
        enhanced = image
        
        if self.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(self.brightness)
        
        if self.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(self.contrast)
        
        if self.sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(self.sharpness)
        
        return enhanced
    
    def reset_enhancements(self):
        self.brightness = 1.0
        self.contrast = 1.0
        self.sharpness = 1.0


# ==================== Main Application ====================
class ImageDistortionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Distortion Viewer")
        self.root.geometry("900x750")
        
        # Initialize components
        self.filter_registry = FilterRegistry()
        self.image_state = ImageState()
        self.view_controller = ViewController()
        self.image_processor = ImageProcessor(self.filter_registry)
        
        # UI Variables
        self.resize_job = None
        self.distortion_type = tk.StringVar(value="Edge Detection")
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.sharpness_var = tk.DoubleVar(value=1.0)
        
        # Display references
        self.tk_original_display = None
        self.tk_distorted_display = None
        
        self.setup_ui()
        self.bind_events()
    
    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # View frame with canvas
        view_frame = ttk.Frame(self.main_frame)
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.columnconfigure(0, weight=1)
        view_frame.rowconfigure(0, weight=1)
        
        self.drop_target = tk.Frame(view_frame, relief="sunken", borderwidth=2)
        self.drop_target.grid(row=0, column=0, sticky="nsew")
        
        self.drop_label = ttk.Label(self.drop_target, text="Drag and Drop an Image Here", 
                                    style="Header.TLabel")
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.canvas = tk.Canvas(self.drop_target, bg="gray20", cursor="arrow")
        
        self.zoom_slider = ttk.Scale(view_frame, from_=self.view_controller.max_zoom * 100, 
                                     to=self.view_controller.min_zoom * 100, orient="vertical")
        self.comparison_slider = ttk.Scale(self.main_frame, from_=0, to=800, orient="horizontal", 
                                          command=self.update_image_view)
        
        # Controls frame
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        
        # Filter selection
        map_options_frame = ttk.LabelFrame(controls_frame, text="Distortion Map Type", padding="10")
        map_options_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        for i, filter_name in enumerate(self.filter_registry.get_filter_names()):
            rb = ttk.Radiobutton(map_options_frame, text=filter_name, 
                               variable=self.distortion_type, value=filter_name, 
                               command=self.regenerate_distortion_map)
            rb.grid(row=i // 4, column=i % 4, sticky="w", padx=5, pady=2)
        
        # Enhancement controls
        adjust_frame = ttk.LabelFrame(controls_frame, text="Distortion Adjustments", padding="10")
        adjust_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        adjust_frame.columnconfigure(1, weight=1)
        
        ttk.Label(adjust_frame, text="Brightness:").grid(row=0, column=0, sticky="w")
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.brightness_var, 
                 orient="horizontal", command=self.on_enhancement_change).grid(row=0, column=1, sticky="ew")
        
        ttk.Label(adjust_frame, text="Contrast:").grid(row=1, column=0, sticky="w")
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.contrast_var, 
                 orient="horizontal", command=self.on_enhancement_change).grid(row=1, column=1, sticky="ew")
        
        ttk.Label(adjust_frame, text="Sharpness:").grid(row=2, column=0, sticky="w")
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.sharpness_var, 
                 orient="horizontal", command=self.on_enhancement_change).grid(row=2, column=1, sticky="ew")
        
        ttk.Button(adjust_frame, text="Reset", command=self.reset_enhancements).grid(
            row=0, column=2, rowspan=3, sticky="ns", padx=10)
        
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
    
    def bind_events(self):
        self.root.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        self.zoom_slider.bind("<ButtonRelease-1>", self.handle_zoom_slider_release)
        
        if platform.system() == "Linux":
            self.canvas.bind("<Button-4>", self.handle_mouse_wheel)
            self.canvas.bind("<Button-5>", self.handle_mouse_wheel)
        else:
            self.canvas.bind("<MouseWheel>", self.handle_mouse_wheel)
        
        self.drop_target.drop_target_register(DND_FILES)
        self.drop_target.dnd_bind('<<Drop>>', self.handle_drop)
    
    def handle_drop(self, event):
        image_path = event.data.strip('{}')
        self.load_and_process_image(image_path)
    
    def load_and_process_image(self, image_path):
        try:
            self.image_state.load_image(image_path)
            self.reset_enhancements()
            self.regenerate_distortion_map(update_view=False)
            self.reset_view_for_new_image()
        except Exception as e:
            self.drop_label.config(text=f"Error opening image: {e}")
    
    def regenerate_distortion_map(self, update_view=True):
        if not self.image_state.has_image():
            return
        
        filter_name = self.distortion_type.get()
        self.image_state.base_distorted_image = self.image_processor.apply_filter(
            self.image_state.original_image, filter_name)
        
        self.apply_enhancements(update_view=update_view)
    
    def on_enhancement_change(self, event=None):
        self.image_processor.brightness = self.brightness_var.get()
        self.image_processor.contrast = self.contrast_var.get()
        self.image_processor.sharpness = self.sharpness_var.get()
        self.apply_enhancements()
    
    def apply_enhancements(self, update_view=True):
        if not self.image_state.base_distorted_image:
            return
        
        self.image_state.enhanced_distorted_image = self.image_processor.apply_enhancements(
            self.image_state.base_distorted_image)
        
        if update_view:
            self.update_image_view()
    
    def reset_enhancements(self):
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.sharpness_var.set(1.0)
        self.image_processor.reset_enhancements()
        self.apply_enhancements()
    
    def reset_view_for_new_image(self):
        if not self.image_state.has_image():
            return
        
        self.drop_label.place_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.zoom_slider.grid(row=0, column=1, sticky="ns")
        self.comparison_slider.grid(row=1, column=0, sticky="ew", pady=5)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(50, self.reset_view_for_new_image)
            return
        
        self.view_controller.reset_for_image(self.image_state.original_image, 
                                            canvas_width, canvas_height)
        
        self.zoom_slider.config(from_=self.view_controller.max_zoom * 100, 
                               to=self.view_controller.min_zoom * 100)
        self.zoom_slider.set(self.view_controller.zoom_level * 100)
        
        self.comparison_slider.config(to=canvas_width)
        self.comparison_slider.set(canvas_width / 2)
        
        self.update_image_view()
    
    def on_resize(self, event=None):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(200, self.handle_resize)
    
    def handle_resize(self):
        if not self.image_state.has_image():
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        new_min_zoom = min(canvas_width / self.image_state.original_image.width, 
                          canvas_height / self.image_state.original_image.height)
        
        self.view_controller.update_min_zoom(new_min_zoom)
        
        self.zoom_slider.config(to=self.view_controller.min_zoom * 100)
        self.zoom_slider.set(self.view_controller.zoom_level * 100)
        
        self.comparison_slider.config(to=canvas_width)
        self.comparison_slider.set(canvas_width / 2)
        
        self.update_image_view()
    
    def update_image_view(self, event=None):
        if not self.image_state.has_image() or not self.image_state.enhanced_distorted_image:
            return
        
        self.canvas.delete("all")
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        slider_pos = int(self.comparison_slider.get())
        
        if not self.view_controller.is_zoomed():
            self._render_fitted_view(canvas_w, canvas_h, slider_pos)
        else:
            self._render_zoomed_view(canvas_w, canvas_h, slider_pos)
    
    def _render_fitted_view(self, canvas_w, canvas_h, slider_pos):
        """Render image at fit-to-screen zoom level"""
        img_w = int(self.image_state.original_image.width * self.view_controller.min_zoom)
        img_h = int(self.image_state.original_image.height * self.view_controller.min_zoom)
        x_offset = (canvas_w - img_w) // 2
        y_offset = (canvas_h - img_h) // 2
        
        original_resized = self.image_state.original_image.resize((img_w, img_h), 
                                                                  Image.Resampling.LANCZOS)
        distorted_resized = self.image_state.enhanced_distorted_image.resize((img_w, img_h), 
                                                                             Image.Resampling.LANCZOS)
        
        clip_pos = slider_pos - x_offset
        
        if clip_pos > 0:
            left = original_resized.crop((0, 0, min(clip_pos, img_w), img_h))
            self.tk_original_display = ImageTk.PhotoImage(left)
            self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_original_display)
        
        if clip_pos < img_w:
            right = distorted_resized.crop((max(0, clip_pos), 0, img_w, img_h))
            self.tk_distorted_display = ImageTk.PhotoImage(right)
            self.canvas.create_image(x_offset + max(0, clip_pos), y_offset, anchor="nw", 
                                    image=self.tk_distorted_display)
        
        if slider_pos >= x_offset and slider_pos <= x_offset + img_w:
            self.canvas.create_line(slider_pos, y_offset, slider_pos, y_offset + img_h, 
                                   fill="red", width=2)
    
    def _render_zoomed_view(self, canvas_w, canvas_h, slider_pos):
        """Render image at zoomed-in level"""
        zoomed_w = self.image_state.original_image.width * self.view_controller.zoom_level
        zoomed_h = self.image_state.original_image.height * self.view_controller.zoom_level
        
        self.view_controller.constrain_offsets(zoomed_w, zoomed_h, canvas_w, canvas_h)
        
        src_x = self.view_controller.view_offset_x / self.view_controller.zoom_level
        src_y = self.view_controller.view_offset_y / self.view_controller.zoom_level
        src_w = canvas_w / self.view_controller.zoom_level
        src_h = canvas_h / self.view_controller.zoom_level
        
        box = (src_x, src_y, src_x + src_w, src_y + src_h)
        
        original_crop = self.image_state.original_image.crop(box).resize((canvas_w, canvas_h), 
                                                                         Image.Resampling.LANCZOS)
        distorted_crop = self.image_state.enhanced_distorted_image.crop(box).resize((canvas_w, canvas_h), 
                                                                                    Image.Resampling.LANCZOS)
        
        left = original_crop.crop((0, 0, slider_pos, canvas_h))
        self.tk_original_display = ImageTk.PhotoImage(left)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_original_display)
        
        if slider_pos < canvas_w:
            right = distorted_crop.crop((slider_pos, 0, canvas_w, canvas_h))
            self.tk_distorted_display = ImageTk.PhotoImage(right)
            self.canvas.create_image(slider_pos, 0, anchor="nw", image=self.tk_distorted_display)
        
        self.canvas.create_line(slider_pos, 0, slider_pos, canvas_h, fill="red", width=2)
    
    def set_zoom(self, new_zoom_level, anchor_x, anchor_y):
        if not self.image_state.has_image():
            return
        
        actual_zoom = self.view_controller.set_zoom(new_zoom_level, anchor_x, anchor_y)
        self.zoom_slider.set(actual_zoom * 100)
        self.update_image_view()
    
    def handle_mouse_wheel(self, event):
        factor = 0.9 if (event.num == 5 or event.delta < 0) else 1.1
        new_zoom = self.view_controller.zoom_level * factor
        self.set_zoom(new_zoom, event.x, event.y)
    
    def handle_zoom_slider_release(self, event=None):
        new_zoom = float(self.zoom_slider.get()) / 100
        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        self.set_zoom(new_zoom, canvas_center_x, canvas_center_y)
    
    def start_pan(self, event):
        if self.view_controller.is_zoomed():
            self.view_controller.start_pan(event.x, event.y)
            self.canvas.config(cursor="fleur")
    
    def pan_image(self, event):
        if self.view_controller.pan(event.x, event.y):
            self.update_image_view()
    
    def end_pan(self, event):
        self.view_controller.end_pan()
        self.canvas.config(cursor="arrow")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageDistortionApp(root)
    root.mainloop()