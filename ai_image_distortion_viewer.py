import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageEnhance, ImageChops
from tkinterdnd2 import DND_FILES, TkinterDnD
import platform
import io

try:
    import numpy as np
    from scipy.fft import fft2, fftshift
except ImportError:
    print("Error: NumPy and SciPy are required for advanced features.")
    print("Please install them using: pip install numpy scipy")
    exit()

class ImageDistortionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Distortion Viewer")
        self.root.geometry("900x750")

        # Image State
        self.original_image = None
        self.base_distorted_image = None
        self.enhanced_distorted_image = None
        self.tk_original_display = None
        self.tk_distorted_display = None
        
        # UI/State Variables
        self.resize_job = None
        self.distortion_type = tk.StringVar(value="Edge Detection")
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.sharpness_var = tk.DoubleVar(value=1.0)

        # Zoom/Pan State
        self.zoom_level = 1.0; self.max_zoom = 10.0; self.min_zoom = 1.0 
        self.view_offset_x = 0; self.view_offset_y = 0
        self.is_panning = False; self.last_drag_x = 0; self.last_drag_y = 0

        self.setup_ui()
        self.bind_events()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1); self.main_frame.rowconfigure(0, weight=1)

        view_frame = ttk.Frame(self.main_frame)
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.columnconfigure(0, weight=1); view_frame.rowconfigure(0, weight=1)

        self.drop_target = tk.Frame(view_frame, relief="sunken", borderwidth=2)
        self.drop_target.grid(row=0, column=0, sticky="nsew")

        self.drop_label = ttk.Label(self.drop_target, text="Drag and Drop an Image Here", style="Header.TLabel")
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        self.canvas = tk.Canvas(self.drop_target, bg="gray20", cursor="arrow")
        
        self.zoom_slider = ttk.Scale(view_frame, from_=self.max_zoom * 100, to=self.min_zoom * 100, orient="vertical")
        self.comparison_slider = ttk.Scale(self.main_frame, from_=0, to=800, orient="horizontal", command=self.update_image_view)
        
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls_frame.columnconfigure(0, weight=1); controls_frame.columnconfigure(1, weight=1)

        map_options_frame = ttk.LabelFrame(controls_frame, text="Distortion Map Type", padding="10")
        map_options_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # --- New: Added ELA 2 (Luminance) to the list ---
        map_options = [
            "Edge Detection", "Color Emboss", "Solarize", "Frequency (FFT)", 
            "Noise Residual", "ELA (Error Level Analysis)", "ELA 2 (Luminance)", "Color Discrepancy"
        ]
        for i, option in enumerate(map_options):
            rb = ttk.Radiobutton(map_options_frame, text=option, variable=self.distortion_type, value=option, command=self.regenerate_distortion_map)
            rb.grid(row=i // 4, column=i % 4, sticky="w", padx=5, pady=2)
        
        adjust_frame = ttk.LabelFrame(controls_frame, text="Distortion Adjustments", padding="10")
        adjust_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        adjust_frame.columnconfigure(1, weight=1)

        ttk.Label(adjust_frame, text="Brightness:").grid(row=0, column=0, sticky="w")
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.brightness_var, orient="horizontal", command=self.apply_enhancements).grid(row=0, column=1, sticky="ew")
        ttk.Label(adjust_frame, text="Contrast:").grid(row=1, column=0, sticky="w")
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.contrast_var, orient="horizontal", command=self.apply_enhancements).grid(row=1, column=1, sticky="ew")
        ttk.Label(adjust_frame, text="Sharpness:").grid(row=2, column=0, sticky="w")
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.sharpness_var, orient="horizontal", command=self.apply_enhancements).grid(row=2, column=1, sticky="ew")
        ttk.Button(adjust_frame, text="Reset", command=self.reset_enhancements).grid(row=0, column=2, rowspan=3, sticky="ns", padx=10)

        style = ttk.Style(); style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))

    def bind_events(self):
        # (This function is identical to the previous version)
        self.root.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.start_pan); self.canvas.bind("<B1-Motion>", self.pan_image); self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        self.zoom_slider.bind("<ButtonRelease-1>", self.handle_zoom_slider_release)
        if platform.system() == "Linux": self.canvas.bind("<Button-4>", self.handle_mouse_wheel); self.canvas.bind("<Button-5>", self.handle_mouse_wheel)
        else: self.canvas.bind("<MouseWheel>", self.handle_mouse_wheel)
        self.drop_target.drop_target_register(DND_FILES); self.drop_target.dnd_bind('<<Drop>>', self.handle_drop)

    def load_and_process_image(self):
        # (This function is identical to the previous version)
        try:
            self.original_image = Image.open(self.image_path).convert("RGB")
            self.reset_enhancements()
            self.regenerate_distortion_map(update_view=False)
            self.reset_view_for_new_image()
        except Exception as e:
            self.drop_label.config(text=f"Error opening image: {e}")

    def regenerate_distortion_map(self, update_view=True):
        if not self.original_image: return
        self.base_distorted_image = self.create_distortion_map(self.original_image)
        self.apply_enhancements(update_view=update_view)
        
    def apply_enhancements(self, event=None, update_view=True):
        # (This function is identical to the previous version)
        if not self.base_distorted_image: return
        enhanced = self.base_distorted_image
        enhancer = ImageEnhance.Brightness(enhanced); enhanced = enhancer.enhance(self.brightness_var.get())
        enhancer = ImageEnhance.Contrast(enhanced); enhanced = enhancer.enhance(self.contrast_var.get())
        enhancer = ImageEnhance.Sharpness(enhanced); enhanced = enhancer.enhance(self.sharpness_var.get())
        self.enhanced_distorted_image = enhanced
        if update_view: self.update_image_view()
        
    def reset_enhancements(self):
        self.brightness_var.set(1.0); self.contrast_var.set(1.0); self.sharpness_var.set(1.0)
        self.apply_enhancements()

    def reset_view_for_new_image(self):
        # (This function is identical to the previous version)
        if not self.original_image: return
        self.drop_label.place_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.zoom_slider.grid(row=0, column=1, sticky="ns")
        self.comparison_slider.grid(row=1, column=0, sticky="ew", pady=5)
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: self.root.after(50, self.reset_view_for_new_image); return
        ratio = min(canvas_width / self.original_image.width, canvas_height / self.original_image.height)
        self.min_zoom = ratio; self.zoom_level = self.min_zoom
        self.zoom_slider.config(from_=self.max_zoom * 100, to=self.min_zoom * 100); self.zoom_slider.set(self.zoom_level * 100)
        self.view_offset_x = 0; self.view_offset_y = 0
        self.comparison_slider.config(to=canvas_width); self.comparison_slider.set(canvas_width / 2)
        self.update_image_view()

    def handle_resize(self):
        # (This function is identical to the previous version)
        if not self.original_image: return
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return
        new_min_zoom = min(canvas_width / self.original_image.width, canvas_height / self.original_image.height)
        self.zoom_level = self.zoom_level if self.zoom_level > self.min_zoom else new_min_zoom
        self.min_zoom = new_min_zoom
        self.zoom_slider.config(to=self.min_zoom * 100); self.zoom_slider.set(self.zoom_level * 100)
        self.comparison_slider.config(to=canvas_width); self.comparison_slider.set(canvas_width / 2)
        self.update_image_view()

    def update_image_view(self, event=None):
        # (This function is identical to the previous version)
        if not self.original_image or not self.enhanced_distorted_image: return
        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width(); canvas_h = self.canvas.winfo_height(); slider_pos = int(self.comparison_slider.get())
        if self.zoom_level <= self.min_zoom:
            img_w = int(self.original_image.width * self.min_zoom); img_h = int(self.original_image.height * self.min_zoom)
            x_offset = (canvas_w - img_w) // 2; y_offset = (canvas_h - img_h) // 2
            original_resized = self.original_image.resize((img_w, img_h), Image.Resampling.LANCZOS)
            distorted_resized = self.enhanced_distorted_image.resize((img_w, img_h), Image.Resampling.LANCZOS)
            clip_pos = slider_pos - x_offset
            if clip_pos > 0:
                left = original_resized.crop((0, 0, min(clip_pos, img_w), img_h)); self.tk_original_display = ImageTk.PhotoImage(left)
                self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_original_display)
            if clip_pos < img_w:
                right = distorted_resized.crop((max(0, clip_pos), 0, img_w, img_h)); self.tk_distorted_display = ImageTk.PhotoImage(right)
                self.canvas.create_image(x_offset + max(0, clip_pos), y_offset, anchor="nw", image=self.tk_distorted_display)
            if slider_pos >= x_offset and slider_pos <= x_offset + img_w: self.canvas.create_line(slider_pos, y_offset, slider_pos, y_offset + img_h, fill="red", width=2)
        else:
            zoomed_w = self.original_image.width * self.zoom_level; zoomed_h = self.original_image.height * self.zoom_level
            self.view_offset_x = max(0, min(self.view_offset_x, zoomed_w - canvas_w)); self.view_offset_y = max(0, min(self.view_offset_y, zoomed_h - canvas_h))
            src_x = self.view_offset_x / self.zoom_level; src_y = self.view_offset_y / self.zoom_level; src_w = canvas_w / self.zoom_level; src_h = canvas_h / self.zoom_level
            box = (src_x, src_y, src_x + src_w, src_y + src_h)
            original_crop = self.original_image.crop(box).resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
            distorted_crop = self.enhanced_distorted_image.crop(box).resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
            left = original_crop.crop((0, 0, slider_pos, canvas_h)); self.tk_original_display = ImageTk.PhotoImage(left)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_original_display)
            if slider_pos < canvas_w:
                right = distorted_crop.crop((slider_pos, 0, canvas_w, canvas_h)); self.tk_distorted_display = ImageTk.PhotoImage(right)
                self.canvas.create_image(slider_pos, 0, anchor="nw", image=self.tk_distorted_display)
            self.canvas.create_line(slider_pos, 0, slider_pos, canvas_h, fill="red", width=2)

    def create_distortion_map(self, image):
        # --- Updated dispatcher with ELA 2 ---
        selection = self.distortion_type.get()
        dispatch = {
            "Edge Detection": self._create_edge_map, "Color Emboss": lambda img: img.filter(ImageFilter.EMBOSS),
            "Solarize": lambda img: ImageOps.solarize(img, threshold=128), "Frequency (FFT)": self._create_fft_map,
            "Noise Residual": self._create_noise_map, "ELA (Error Level Analysis)": self._create_ela_map, 
            "ELA 2 (Luminance)": self._create_ela_luminance_map, "Color Discrepancy": self._create_color_std_map
        }
        return dispatch.get(selection, self._create_edge_map)(image)

    def _create_edge_map(self, image):
        # (Identical to previous version)
        distorted = ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES)
        return ImageOps.autocontrast(distorted, cutoff=5).convert("RGB")
    
    def _create_fft_map(self, image):
        # (Identical to previous version)
        gray_image = image.convert("L"); np_image = np.array(gray_image)
        f_transform = fft2(np_image); f_transform_shifted = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
        magnitude_spectrum = 255 * (magnitude_spectrum / np.max(magnitude_spectrum))
        return Image.fromarray(magnitude_spectrum.astype(np.uint8)).convert("RGB")

    def _create_noise_map(self, image):
        # (Identical to previous version)
        denoised = image.filter(ImageFilter.GaussianBlur(radius=1))
        return ImageChops.subtract(image, denoised, scale=2.0, offset=128).convert("RGB")

    def _create_ela_map(self, image):
        # (Identical to previous version)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=95); buffer.seek(0)
        resaved_image = Image.open(buffer)
        diff = ImageChops.difference(image, resaved_image)
        return ImageOps.autocontrast(diff, cutoff=2).convert("RGB")
    
    # --- New ELA 2 Implementation ---
    def _create_ela_luminance_map(self, image):
        # Work with the luminance channel for a cleaner signal
        original_lum = image.convert("L")
        
        buffer = io.BytesIO()
        original_lum.save(buffer, "JPEG", quality=95); buffer.seek(0)
        resaved_lum = Image.open(buffer)

        diff = ImageChops.difference(original_lum, resaved_lum)
        return ImageOps.autocontrast(diff, cutoff=2).convert("RGB")

    def _create_color_std_map(self, image):
        # (Identical to previous version)
        np_image = np.array(image, dtype=np.float32)
        std_map = np.std(np_image, axis=2)
        std_map = 255 * (std_map / np.max(std_map) if np.max(std_map) > 0 else 0)
        return Image.fromarray(std_map.astype(np.uint8)).convert("RGB")

    # --- Other Event Handlers (Unchanged) ---
    def handle_drop(self, event): self.image_path = event.data.strip('{}'); self.load_and_process_image()
    def on_resize(self, event=None):
        if self.resize_job: self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(200, self.handle_resize)
    def set_zoom(self, new_zoom_level, anchor_x, anchor_y):
        if not self.original_image: return
        old_zoom = self.zoom_level
        self.zoom_level = max(self.min_zoom, min(new_zoom_level, self.max_zoom))
        self.view_offset_x = (self.view_offset_x + anchor_x) * (self.zoom_level / old_zoom) - anchor_x
        self.view_offset_y = (self.view_offset_y + anchor_y) * (self.zoom_level / old_zoom) - anchor_y
        self.zoom_slider.set(self.zoom_level * 100); self.update_image_view()
    def handle_mouse_wheel(self, event):
        factor = 0.9 if (event.num == 5 or event.delta < 0) else 1.1
        self.set_zoom(self.zoom_level * factor, event.x, event.y)
    def handle_zoom_slider_release(self, event=None):
        new_zoom = float(self.zoom_slider.get()) / 100
        self.set_zoom(new_zoom, self.canvas.winfo_width()/2, self.canvas.winfo_height()/2)
    def start_pan(self, event):
        if self.zoom_level > self.min_zoom: self.is_panning = True; self.last_drag_x = event.x; self.last_drag_y = event.y; self.canvas.config(cursor="fleur")
    def pan_image(self, event):
        if self.is_panning:
            dx = event.x - self.last_drag_x; dy = event.y - self.last_drag_y
            self.view_offset_x -= dx; self.view_offset_y -= dy
            self.last_drag_x = event.x; self.last_drag_y = event.y
            self.update_image_view()
    def end_pan(self, event): self.is_panning = False; self.canvas.config(cursor="arrow")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageDistortionApp(root)
    root.mainloop()