import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QComboBox, QGroupBox, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Matplotlib setup for PyQt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# NIBabel for loading medical images
import nibabel as nib

# Main logic class that contains static methods for reconstruction calculation and applying noise
class MRILogic:
    @staticmethod
    # Reconstructs a 2D slice using its full k-space data
    def reconstruct_2d(data_slice):
        k_space = np.fft.fft2(data_slice)              # Converts data from pixel space to k-space (frequencies) using a 2D Fast Fourier Transform
        k_space_shifted = np.fft.fftshift(k_space)     # Shifts brightest part (0,0 frequency) into the center of the matrix for visualization
        reconstruction = np.fft.ifft2(np.fft.ifftshift(k_space_shifted)) # Converts shifted k-space back to spaital domain (Image pixels data) using inverse fourier
        reconstruction_magnitude = np.abs(reconstruction) # Generates a single array of real magnitude pixel values from the complex numbers, needed for viewing the image
        return k_space_shifted, reconstruction_magnitude

    @staticmethod
    # Same as 2D reconstruction but it undersamples the k-space with respect to acceleration factor
    def reconstruct_2d_aliased(data_slice, acceleration_factor = 2):
        k_full_shifted = np.fft.fftshift(np.fft.fft2(data_slice))
        k_aliased_shifted = np.zeros_like(k_full_shifted) # New k-space with same size as input k-space, filled with zeros
        k_aliased_shifted[ : : acceleration_factor, :] = k_full_shifted[ : : acceleration_factor, :] # Copy every second row to new k-space, leaving rows (1, 3, 5, ..) zeros
        reconstruction_aliased = np.fft.ifft2(np.fft.ifftshift(k_aliased_shifted))
        reconstruction_aliased_magnitude = np.abs(reconstruction_aliased)
        return k_aliased_shifted, reconstruction_aliased_magnitude

    @staticmethod
    # Reconstructs whole 3D volume
    def reconstruct_3d(volume):
        k_space = np.fft.fftn(volume) # Performs 3D FFT to convert 3D volume to 3D k-space
        k_space_shifted = np.fft.fftshift(k_space)
        reconstruction = np.fft.ifftn(np.fft.ifftshift(k_space_shifted))
        reconstruction_magnitude = np.abs(reconstruction)
        return k_space_shifted, reconstruction_magnitude

    @staticmethod
    # Simulates adding gaussian noise to the data
    def apply_noise(base_volume, noise_level_percent):
        if base_volume is None: 
            return None
        signal_range = base_volume.max() - base_volume.min()    # Brightest - Darkest pixel in the volume
        std_dev = (noise_level_percent / 100.0) * signal_range  # Scales standard deviation relative to our signal range
        noise = np.random.normal(0, std_dev, base_volume.shape) # Generates an array of same shape as our volume, containing gaussian noise
        noisy_volume = base_volume + noise
        return noisy_volume

    @staticmethod
    # Calculates signal to noise ratio for a single 2D slice
    def calculate_snr(image_slice):
        height, width = image_slice.shape
        if height < 40 or width < 40: # Image is too small to draw a ROI
            return 0
        tissue_roi = image_slice[height // 2 - 10 : height // 2 + 10, width // 2 - 10 : width // 2 + 10] # Slices out a 20 * 20 ROI from the center of our image to indicate signal
        air_roi = image_slice[0 : 30, 0 : 30] # Slices out a 30 * 30 ROI from the top left of our image to indicate background air (noise)
        noise_std = np.std(air_roi) + 1e-12  # Calculates standard deviation for all pixel values in our background ROI, then adds a small amount to each value to avoid zeros
        signal_mean = np.mean(tissue_roi)
        snr = signal_mean / noise_std
        return max(0, snr)

# Canvas Helper Class
class SimpleCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, facecolor="#0b1220"):
        fig = Figure(figsize=(width, height), facecolor=facecolor)
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#000000')
        super(SimpleCanvas, self).__init__(fig)
        self.setParent(parent)
        # Prefer expanding so layouts can give more space to canvases
        try:
            from PyQt5.QtWidgets import QSizePolicy
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

    def plot(self, data, title="", cmap='gray'):
        self.axes.clear()
        self.axes.imshow(data, cmap=cmap, interpolation='bilinear')
        self.axes.set_title(title, color='white', fontsize=14)
        self.axes.axis('off')
        self.draw()

# Main Application Window
class FFTDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI 2D vs 3D FFT Demo")
        self.setGeometry(100, 100, 1400, 800)

        # --- Data Placeholders (State) ---
        self.base_volume = None
        self.noisy_volume = None
        self.kspace_2d = None
        self.kspace_3d = None
        self.reconstruction_2d_norm = None
        self.reconstruction_3d_norm = None
        self.reconstruction_3d_raw = None
        self.snr_2d = None
        self.snr_3d = None
        self.current_reconstruction_mode = "None"

        # --- UI State Parameters ---
        self.slice_idx = 0
        self.resolution_x = 64
        self.resolution_y = 64
        self.slices_num = 50                       
        
        # --- MRI Physics Parameters (State) ---
        self.FOV_x_mm = 181.0
        self.FOV_y_mm = 217.0
        self.base_thickness_mm = 1.0
        self.current_2d_thickness_slices = 5
        self.TR_2D = 0.3
        self.TR_3D = 0.02
        self.averages = 2
        self.noise_level_percent = 0

        # --- Build the UI ---
        self.generate_volume()
        self._build_ui()
        self._setup_styles()
        
        self.update_display()
        self.on_thick_changed(self.current_2d_thickness_slices)
        
    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Left Panel: Controls ---
        left_panel_widget = QWidget()
        controls_panel_layout = QVBoxLayout(left_panel_widget)
        # increase spacing between widgets in the sidebar for better separation
        controls_panel_layout.setSpacing(32)
        # give some content margins so elements are not flush to the container edges
        controls_panel_layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("MRI Reconstruction Controls")
        header.setFont(QFont('Arial', 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        controls_panel_layout.addWidget(header)

        # Group 1: Data & Slice Info
        data_group = QGroupBox("Data & Slice")
        data_layout = QGridLayout()
        data_layout.setSpacing(16)
        data_layout.setContentsMargins(30, 30, 30, 30)
        self.btn_gen = QPushButton("Reload Volume")
        self.btn_gen.clicked.connect(self.on_generate)
        self.btn_gen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) 
        data_layout.addWidget(self.btn_gen, 0, 0, 1, 2)
        self.slice_label = QLabel(f'Slice: {self.slice_idx}')
        data_layout.addWidget(self.slice_label, 1, 0)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.slices_num - 1)
        self.slice_slider.setValue(self.slice_idx)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        data_layout.addWidget(self.slice_slider, 1, 1)
        data_group.setLayout(data_layout)
        controls_panel_layout.addWidget(data_group)
        
        controls_panel_layout.addStretch(1)

        # Group 2: Scan Parameters
        params_group = QGroupBox("Scan Parameters")
        params_layout = QGridLayout()
        params_layout.setSpacing(16)
        params_layout.setContentsMargins(30, 30, 30, 30)
        self.thick_label = QLabel(f'Sim. 2D Thickness: {self.current_2d_thickness_slices} slices')
        params_layout.addWidget(self.thick_label, 0, 0)
        self.thick_slider = QSlider(Qt.Horizontal)
        self.thick_slider.setMinimum(1)
        self.thick_slider.setMaximum(30)
        self.thick_slider.setValue(self.current_2d_thickness_slices)
        self.thick_slider.valueChanged.connect(self.on_thick_changed)
        params_layout.addWidget(self.thick_slider, 0, 1)
        self.noise_label = QLabel(f"Added Noise: {self.noise_level_percent}%")
        params_layout.addWidget(self.noise_label, 1, 0)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(50)
        self.noise_slider.setValue(self.noise_level_percent)
        self.noise_slider.valueChanged.connect(self.on_noise_changed)
        params_layout.addWidget(self.noise_slider, 1, 1)
        params_layout.addWidget(QLabel('Colormap:'), 2, 0)
        self.cmap_box = QComboBox()
        self.cmap_box.addItems(['gray', 'magma', 'viridis', 'inferno'])
        self.cmap_box.setCurrentText('gray')
        self.cmap_box.currentTextChanged.connect(self.update_display)
        params_layout.addWidget(self.cmap_box, 2, 1)
        params_group.setLayout(params_layout)
        controls_panel_layout.addWidget(params_group)
        
        controls_panel_layout.addStretch(1)

        # Group 3: Reconstruction Buttons
        recon_group = QGroupBox("Reconstruction")
        recon_layout = QVBoxLayout()
        # slightly increase spacing inside the reconstruction buttons group
        recon_layout.setSpacing(18)
        recon_layout.setContentsMargins(30, 30, 30, 30)
        self.btn_fft2d = QPushButton("Run 2D FFT (Sim. Thick Slice)")
        self.btn_fft2d.clicked.connect(self.on_fft2d)
        recon_layout.addWidget(self.btn_fft2d)
        self.btn_fft3d = QPushButton("Run 3D FFT (Full Volume)")
        self.btn_fft3d.clicked.connect(self.on_fft3d)
        recon_layout.addWidget(self.btn_fft3d)
        self.btn_fft2d_alias = QPushButton("Run 2D FFT + Aliasing (2x Faster)")
        self.btn_fft2d_alias.setObjectName("AliasButton")
        self.btn_fft2d_alias.clicked.connect(self.on_fft2d_alias)
        recon_layout.addWidget(self.btn_fft2d_alias)
        recon_group.setLayout(recon_layout)
        controls_panel_layout.addWidget(recon_group)
        
        controls_panel_layout.addStretch(1)

        # Group 4: Info & Status
        info_group = QGroupBox("Info & Status")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(16)
        info_layout.setContentsMargins(30, 30, 30, 30)
        self.info_2d_thick = QLabel("2D Physical Thickness: -")
        self.info_3d_thick = QLabel("3D Effective Thickness: -")
        self.voxel_size_label = QLabel("Voxel Size (2D): -")
        self.voxel_size_label_3d = QLabel("Voxel Size (3D): -")
        self.scan_time_label = QLabel("Scan Time 2D/3D: -")
        self.info = QLabel("Status: Ready")
        info_layout.addWidget(self.info_2d_thick)
        info_layout.addWidget(self.info_3d_thick)
        info_layout.addWidget(self.voxel_size_label)
        info_layout.addWidget(self.voxel_size_label_3d)
        info_layout.addWidget(self.scan_time_label)
        info_layout.addWidget(self.info)
        info_group.setLayout(info_layout)
        controls_panel_layout.addWidget(info_group)
        
        controls_panel_layout.addStretch(2)

        # Set a smaller minimum width on the left panel so the canvases can be larger
        left_panel_widget.setMinimumWidth(300)
        # Give the left panel a smaller stretch and the right panel (canvases) a larger stretch
        main_layout.addWidget(left_panel_widget, 1) # Panel takes 1 part

        # Right Panel: (2 * 2 Grid)
        right_panel_widget = QWidget()
        canvas_panel_layout = QVBoxLayout(right_panel_widget)
        canvas_panel_layout.setSpacing(10)

        # Top Row: Input + Reconstructed Images
        top_row_layout = QHBoxLayout()
        # Larger canvas defaults so images render bigger by default
        self.img_input = SimpleCanvas(self, width=7, height=7)
        self.img_reconstructed = SimpleCanvas(self, width=7, height=7)
        top_row_layout.addWidget(self.img_input, 1)
        top_row_layout.addWidget(self.img_reconstructed, 1)

        canvas_panel_layout.addLayout(top_row_layout, 1)

        # Bottom Row: Input + Reconstructed K-Space
        bottom_row_layout = QHBoxLayout()
        self.canvas_k_space_input = SimpleCanvas(self, width=7, height=7)
        self.canvas_k_space_reconstructed = SimpleCanvas(self, width=7, height=7)
        bottom_row_layout.addWidget(self.canvas_k_space_input, 1)
        bottom_row_layout.addWidget(self.canvas_k_space_reconstructed, 1)

        canvas_panel_layout.addLayout(bottom_row_layout, 1)

        main_layout.addWidget(right_panel_widget, 4)

    def _setup_styles(self):
        """Applies a global CSS stylesheet"""
        self.setStyleSheet("""
            /* Modern dark theme with sky-blue accents */
            QMainWindow, QWidget {
                background-color: #0f1724; /* very dark bluish */
                color: #e6eef8; /* soft light text */
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                background-color: #0b1220; /* slightly lighter panel */
                border: 1px solid #132033; /* subtle border */
                border-radius: 8px; margin-top: 12px;
                font-size: 18px;
                font-weight: 700;
                padding: 6px; /* small padding inside the box */
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 6px 10px; /* move title off the edge */
                color: #7dd3fc; /* sky-blue accent */
                font-weight: 800;
            }
            QLabel { 
                font-size: 14px; 
                color: #cfe8ff; 
            }
            QPushButton {
                background-color: #06b6d4; /* teal/cyan */
                color: #071423; /* near-black text for contrast */
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #04a2b0; }
            QPushButton:pressed { background-color: #038188; }
            #AliasButton { background-color: #ef4444; color: white; }
            #AliasButton:hover { background-color: #dc2626; }
            #AliasButton:pressed { background-color: #b91c1c; }
            QSlider::groove:horizontal {
                border: 1px solid #132033; height: 8px;
                background: #132033; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #7dd3fc; border: 1px solid #7dd3fc;
                width: 20px; height: 20px;
                margin: -6px 0; border-radius: 10px;
            }
            QComboBox {
                background-color: #071423; color: #e6eef8; border: 1px solid #132033;
                padding: 8px; border-radius: 6px; font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background-color: #071423; color: #e6eef8; 
                selection-background-color: #06b6d4;
            }
            /* Make minor UI elements more muted */
            QLabel#voxel_size_label, QLabel#voxel_size_label_3d { color: #ffd580; }
        """)
        
        # Apply special styles for info labels
        for label in [self.info_2d_thick, self.info_3d_thick, self.voxel_size_label, self.voxel_size_label_3d, self.scan_time_label, self.info]:
            label.setFont(QFont('Arial', 15, QFont.Bold))

        self.info_2d_thick.setStyleSheet("color: #34E0A1;")
        self.info_3d_thick.setStyleSheet("color: #34A1E0;")
        self.voxel_size_label.setStyleSheet("color: #E0A134;")
        self.voxel_size_label_3d.setStyleSheet("color: #E0A134;")
        self.scan_time_label.setStyleSheet("color: #C875C4;")
        self.info.setStyleSheet("color: #E0E034;")


    # Core Logic Methods
    def compute_voxel_size_mm(self, mode="2D"):
        if self.resolution_x == 0 or self.resolution_y == 0: 
            return 0, 0, 0
        vx = self.FOV_x_mm / self.resolution_x      
        vy = self.FOV_y_mm / self.resolution_y
        if (mode == "2D"): 
            vz = self.current_2d_thickness_slices * self.base_thickness_mm
        else:
            vz = self.base_thickness_mm
        return vx, vy, vz
        
    def compute_scan_time(self, mode="2D"):
        Ny = self.resolution_y
        Nz = self.slices_num
        if mode == "2D":
            time = self.TR_2D * Ny * (Nz / self.current_2d_thickness_slices) * self.averages
            self.scan_time_2d = time
        else:
            time = self.TR_3D * Ny * Nz * self.averages
            self.scan_time_3d = time
        
    def generate_volume(self):
        try:
            img = nib.load('./datasets/t1_icbm_normal_1mm_pn3_rf20.mnc')
            self.base_volume = img.get_fdata()
            self.base_thickness_mm = img.header.get_zooms()[0]
            self.slices_num, self.resolution_y, self.resolution_x = self.base_volume.shape
            
            if hasattr(self, 'slice_slider'):
                 self.slice_slider.setMaximum(self.slices_num - 1)
            
            self._apply_noise()
            print(f"Volume loaded. Shape: {self.base_volume.shape}, Base Thickness: {self.base_thickness_mm}mm")

        except Exception as e:
             print(f"Error loading volume: {e}. Creating dummy data.")
             self.base_volume = np.random.rand(50, 64, 64)
             self.slices_num, self.resolution_y, self.resolution_x = self.base_volume.shape
             self.base_thickness_mm = 1.0
             self._apply_noise()
             if hasattr(self, 'slice_slider'):
                 self.slice_slider.setMaximum(self.slices_num - 1)
        
        self.kspace_2d = self.reconstruction_2d_norm = self.kspace_3d = self.reconstruction_3d_norm = None

    def _apply_noise(self):
        self.noisy_volume = MRILogic.apply_noise(self.base_volume, self.noise_level_percent)
        print(f"Noise level: {self.noise_level_percent}%")

    def _get_thick_slice(self):
        if self.noisy_volume is None: 
            # Return a blank array if no data, to avoid crash on launch
            return np.zeros((self.resolution_y, self.resolution_x))
        end_slice = min(self.slice_idx + self.current_2d_thickness_slices, self.slices_num)
        if self.slice_idx >= end_slice:
             return self.noisy_volume[self.slice_idx, :, :]
        else:
             return np.mean(self.noisy_volume[self.slice_idx : end_slice, :, :], axis=0)
    
    def _get_clean_thick_slice(self):
        if self.base_volume is None: 
            return np.zeros((self.resolution_y, self.resolution_x))
        end_slice = min(self.slice_idx + self.current_2d_thickness_slices, self.slices_num)
        if self.slice_idx >= end_slice:
             return self.base_volume[self.slice_idx, :, :]
        else:
             return np.mean(self.base_volume[self.slice_idx : end_slice, :, :], axis=0)

    # Event Handlers
    def on_generate(self):
        self.generate_volume()
        self.update_display()
        self.info.setText("Status: Volume reloaded.")

    def on_slice_changed(self, val):
        self.slice_idx = int(val)
        self.slice_label.setText(f"Slice: {self.slice_idx}")
        self.update_display()

    def on_noise_changed(self, val):
        self.noise_level_percent = int(val)
        self.noise_label.setText(f"Added Noise: {self.noise_level_percent}%")
        self._apply_noise()
        self._clear_all_data()
        self.update_display()
        self.info.setText(f"Status: Noise set to {val}%.")

    def on_thick_changed(self, val):
        self.current_2d_thickness_slices = int(val)
        self.thick_label.setText(f"Sim. 2D Thickness: {val} slices")
        self.update_info_labels()
        
    def on_fft2d(self):
        self.current_reconstruction_mode = "2D"
        self._clear_3d_data()
        data_slice = self._get_thick_slice()
        if data_slice is None: return

        k, recon = MRILogic.reconstruct_2d(data_slice)
        
        self.kspace_2d = k
        self.snr_2d = MRILogic.calculate_snr(recon)
        self.reconstruction_2d_norm = self._normalize(recon)

        self.info.setText(f"Status: 2D FFT done. SNR: {self.snr_2d:.1f}")
        self.update_display()
        
    def on_fft2d_alias(self):
        self.current_reconstruction_mode = "Alias"
        self._clear_3d_data()
        self.snr_2d = None
        data_slice = self._get_thick_slice()
        if data_slice is None: return

        k_aliased, recon_aliased = MRILogic.reconstruct_2d_aliased(data_slice)
        
        self.kspace_2d = k_aliased
        self.reconstruction_2d_norm = self._normalize(recon_aliased)

        self.info.setText(f"Status: 2D FFT with Aliasing (acceleration factor = 2)")
        self.update_display()

    def on_fft3d(self):
        self.current_reconstruction_mode = "3D"
        self._clear_2d_data()
        if self.noisy_volume is None: return

        self.info.setText("Status: Running 3D FFT... please wait...")
        QApplication.processEvents()

        k, recon = MRILogic.reconstruct_3d(self.noisy_volume)

        self.kspace_3d = k
        self.reconstruction_3d_raw = recon
        self.reconstruction_3d_norm = self._normalize(recon)
        self.snr_3d = MRILogic.calculate_snr(recon[self.slice_idx])
        
        self.info.setText(f"Status: 3D FFT done. SNR: {self.snr_3d:.1f}")
        self.update_display()

    # Visualization & Helpers    
    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min() + 1e-12)

    def _clear_all_data(self):
        self._clear_2d_data()
        self._clear_3d_data()

    def _clear_2d_data(self):
        self.kspace_2d = self.reconstruction_2d_norm = self.snr_2d = None

    def _clear_3d_data(self):
        self.kspace_3d = self.reconstruction_3d_norm = self.snr_3d = self.reconstruction_3d_raw = None

    def update_display(self):
        """The main plotting function. Redraws all 4 canvases."""
        cmap = self.cmap_box.currentText()
        
        # Plot Input Phantom Image (Top-Left)
        if self.base_volume is not None:
            img = self.base_volume[self.slice_idx, :, :]
            self.img_input.plot(self._normalize(img), title=f"Input Phantom (Slice {self.slice_idx})", cmap=cmap)
        else:
             self.img_input.plot(np.zeros((self.resolution_y, self.resolution_x)), title="Input Phantom", cmap=cmap)
        
        # Plot Reconstructed Image (Top-Right)
        if self.reconstruction_2d_norm is not None and self.current_reconstruction_mode in ["2D", "Alias"]:
            if self.current_reconstruction_mode == "Alias":
                title = "2D Reconstruction (Aliased)"
            else:
                title = f"2D Reconstruction | SNR: {self.snr_2d:.1f}"
            self.img_reconstructed.plot(self.reconstruction_2d_norm, title=title, cmap=cmap)
        elif self.reconstruction_3d_norm is not None and self.current_reconstruction_mode == "3D":
            snr_current_slice = MRILogic.calculate_snr(self.reconstruction_3d_raw[self.slice_idx])
            title = f"3D Reconstruction | SNR: {snr_current_slice:.1f}"
            self.img_reconstructed.plot(self.reconstruction_3d_norm[self.slice_idx, :, :], title=title, cmap=cmap)
        else:
            self.img_reconstructed.plot(np.zeros((self.resolution_y, self.resolution_x)), title="Reconstruction", cmap=cmap)

        # Plot Input K-Space (Bottom-Left) for the current input slice
        input_slice_data = self._get_clean_thick_slice()
        k_input_shifted, _ = MRILogic.reconstruct_2d(input_slice_data)
        mag_input = np.log(1 + np.abs(k_input_shifted))
        self.canvas_k_space_input.plot(self._normalize(mag_input), title="Input K-Space (Full)", cmap=cmap)

        # Plot Reconstructed K-Space (Bottom-Right)
        if self.kspace_2d is not None and self.current_reconstruction_mode in ["2D", "Alias"]:
            mag = np.log(1 + np.abs(self.kspace_2d))
            k_title = "Reconstructed K-Space (2D)" + (" (Undersampled)" if self.current_reconstruction_mode == "Alias" else "")
            self.canvas_k_space_reconstructed.plot(self._normalize(mag), title=k_title, cmap=cmap)
        elif self.kspace_3d is not None and self.current_reconstruction_mode == "3D":
            mag = np.log(1 + np.abs(self.kspace_3d[self.slice_idx, :, :]))
            self.canvas_k_space_reconstructed.plot(self._normalize(mag), title=f"Reconstructed K-Space (3D Slice {self.slice_idx})", cmap=cmap)
        else:
            self.canvas_k_space_reconstructed.plot(np.zeros((self.resolution_y, self.resolution_x)), title="Reconstruction K-Space", cmap=cmap)

    def update_info_labels(self):
        if self.base_volume is None: return
        try:
            self.compute_scan_time(mode="2D")
            self.compute_scan_time(mode="3D")
            
            t2m = f"{self.scan_time_2d / 60:.1f}m"
            t3m = f"{self.scan_time_3d / 60:.1f}m"
            self.scan_time_label.setText(f"Scan Time 2D/3D: {t2m} / {t3m}")

            vx2, vy2, vz2 = self.compute_voxel_size_mm(mode="2D")
            vx3, vy3, vz3 = self.compute_voxel_size_mm(mode="3D")
            
            self.voxel_size_label.setText(f"Voxel (2D): {vx2:.2f} x {vx2:.2f} x {vz2:.2f} mm")
            self.voxel_size_label_3d.setText(f"Voxel (3D): {vx3:.2f} x {vy3:.2f} x {vz3:.2f} mm")
            self.info_3d_thick.setText(f"3D Effective Thickness: {self.base_thickness_mm:.1f} mm")
            self.info_2d_thick.setText(f"2D Physical Thickness: {vz2:.1f} mm")
            
        except Exception as e:
            print(f"Error updating info labels: {e}")

# application entry point
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FFTDemo()
    win.show()
    sys.exit(app.exec_())