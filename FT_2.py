import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QComboBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import nibabel as nib


# ---------------------------------------------------------------------
# Canvas helper
# ---------------------------------------------------------------------
class SimpleCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=4, height=4, facecolor="#0b1220"):
        fig = Figure(figsize=(width, height), facecolor=facecolor)
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#000000')
        super(SimpleCanvas, self).__init__(fig)

    def plot(self, data, title="", cmap='gray', overlay=None):
        self.axes.clear()
        self.axes.imshow(data, cmap=cmap, interpolation='bilinear')
        if overlay is not None:
            try:
                ov = np.zeros((overlay.shape[0], overlay.shape[1], 4), dtype=float)
                ov[..., 0] = overlay.astype(float)
                ov[..., 3] = overlay.astype(float) * 0.6
                self.axes.imshow(ov, interpolation='nearest')
            except Exception:
                self.axes.contour(overlay, colors='r', linewidths=0.5)
        self.axes.set_title(title, color='white', fontsize=10)
        self.axes.axis('off')
        self.draw()

# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------
class FFTDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D vs 3D FFT MRI Phantom - Comparison")
        self.setGeometry(100, 100, 1000, 600)

        # Parameters ----------------------------------------------------
        
        self.slice_idx = 0  
        self.resolution_x = 0
        self.resolution_y = 0
        self.nslices = 0                           

        # MRI-like physical parameters
        # Practically, FOV is choosed clinically but our data resolution is 1*1*1 mm so we set FOV as the slice shape
        # FOV = DELTA_X * Nx
        self.FOV_x_mm = 181.0           # Field of view (mm)  
        self.FOV_y_mm = 217.0           # Field of view (mm)
        
        # NEW: Base thickness of the loaded 3D volume (native resolution)
        self.base_thickness_mm = 1.0   
        # MODIFIED: This will now be dynamic based on the slider
        self.current_2d_thickness_slices = 5 

        self.TR_2D = 0.3              # repetition time (sec) as the data is T1 image (assumption)
        self.TR_3D = 0.02              # repetition time (sec) (assumption)
        self.noise_level_percent = 0   # Added noise percentage

        # Data placeholders
        self.base_volume = None        # The original, clean volume
        self.noisy_volume = None       # Volume with added noise
        self.kspace_2d = None
        self.recon_2d = None
        self.snr_2d = None
        self.kspace_3d = None
        self.recon_3d = None
        self.snr_2d = None
        self.snr_3d = None
        self.scan_time_2d = None
        self.scan_time_3d = None
        self.averages = 2          # number of capturing all the scan (assumption)

        # Build UI and generate volume
        self.generate_volume()
        self._build_ui()
        self.update_display()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        header = QLabel("Compare 2D/3D FFT Reconstruction (with SNR & Scan Time)")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # --- Canvases -------------------------------------------------
        top = QHBoxLayout()
        g1 = QGroupBox('Phantom Slice')
        g2 = QGroupBox('K-space')
        g3 = QGroupBox('Reconstruction')

        self.canvas_img = SimpleCanvas(self, width=4, height=4)
        self.canvas_k = SimpleCanvas(self, width=4, height=4)
        self.canvas_recon = SimpleCanvas(self, width=4, height=4)

        v1, v2, v3 = QVBoxLayout(g1), QVBoxLayout(g2), QVBoxLayout(g3)
        v1.addWidget(self.canvas_img)
        v2.addWidget(self.canvas_k)
        v3.addWidget(self.canvas_recon)
        top.addWidget(g1)
        top.addWidget(g2)
        top.addWidget(g3)
        layout.addLayout(top)

        # --- Controls -------------------------------------------------
        ctrl_box = QGroupBox('Controls')
        ctrl_layout = QGridLayout()

        # Regenerate button
        self.btn_gen = QPushButton("Reload Volume")
        self.btn_gen.clicked.connect(self.on_generate)
        ctrl_layout.addWidget(self.btn_gen, 0, 0, 1, 2)

        # Slice control
        self.slice_label = QLabel('Slice: 0')
        ctrl_layout.addWidget(self.slice_label, 1, 0)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        # Maximum will be set in generate_volume after we know nslices
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        ctrl_layout.addWidget(self.slice_slider, 1, 1)

        # --- NEW: 2D Slice Thickness Simulation Slider ---
        self.thick_label = QLabel(f'Simulated 2D Thickness: {self.current_2d_thickness_slices} slices')
        ctrl_layout.addWidget(self.thick_label, 2, 0)
        
        self.thick_slider = QSlider(Qt.Horizontal)
        self.thick_slider.setMinimum(1)
        self.thick_slider.setMaximum(30) # Allow averaging up to 30 slices
        self.thick_slider.setValue(self.current_2d_thickness_slices)
        self.thick_slider.valueChanged.connect(self.on_thick_changed)
        ctrl_layout.addWidget(self.thick_slider, 2, 1)

        # --- NEW: Thickness Comparison Display ---
        # Shows fixed 3D effective thickness vs variable 2D physical thickness
        self.info_3d_thick = QLabel("3D Effective Thickness: -")
        self.info_3d_thick.setStyleSheet("color: blue; font-weight: bold;")
        ctrl_layout.addWidget(self.info_3d_thick, 3, 0)
        
        self.info_2d_thick = QLabel("2D Physical Thickness: -")
        self.info_2d_thick.setStyleSheet("color: green; font-weight: bold;")
        ctrl_layout.addWidget(self.info_2d_thick, 3, 1)

        # Colormap
        ctrl_layout.addWidget(QLabel('Colormap:'), 4, 0)
        self.cmap_box = QComboBox()
        self.cmap_box.addItems(['gray', 'magma', 'viridis', 'inferno'])
        self.cmap_box.setCurrentText('gray')
        self.cmap_box.currentTextChanged.connect(self.update_display)
        ctrl_layout.addWidget(self.cmap_box, 4, 1)

        # --- NEW: Noise Slider ---
        self.noise_label = QLabel(f"Added Noise: {self.noise_level_percent}%")
        ctrl_layout.addWidget(self.noise_label, 5, 0)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(50)
        self.noise_slider.setValue(self.noise_level_percent)
        self.noise_slider.valueChanged.connect(self.on_noise_changed)
        ctrl_layout.addWidget(self.noise_slider, 5, 1)

        # FFT buttons
        self.btn_fft2d = QPushButton("2D FFT (Simulated Thick Slice)")
        self.btn_fft2d.clicked.connect(self.on_fft2d)
        ctrl_layout.addWidget(self.btn_fft2d, 6, 0)
        self.btn_fft3d = QPushButton("3D FFT (Full Volume)")
        self.btn_fft3d.clicked.connect(self.on_fft3d)
        ctrl_layout.addWidget(self.btn_fft3d, 6, 1)

        # Scan time and voxel size display ---
        self.scan_time_label = QLabel("Scan Time: -")
        self.scan_time_label.setStyleSheet("color: purple;")
        ctrl_layout.addWidget(self.scan_time_label, 7, 0)

        self.voxel_size_label = QLabel("Voxel Size (mm): - x - x -")
        self.voxel_size_label.setStyleSheet("color: darkorange;")
        ctrl_layout.addWidget(self.voxel_size_label, 7, 1)

        ctrl_box.setLayout(ctrl_layout)
        layout.addWidget(ctrl_box)

        # Info
        self.info = QLabel("Status: ready")
        layout.addWidget(self.info)

        # Initial update of thickness labels
        self.on_thick_changed(self.current_2d_thickness_slices)

    # -----------------------------------------------------------------
    # Core logic
    # -----------------------------------------------------------------
    def compute_voxel_size_mm(self, mode="2D"):
        # in-plane resolution
        vx = self.FOV_x_mm / self.resolution_x      
        vy = self.FOV_y_mm / self.resolution_y
        # Through-plane resolution
        if(mode == "2D"): 
            vz = self.current_2d_thickness_slices  
        else:
            vz = self.base_thickness_mm
        return vx, vy, vz
        
    
    def compute_scan_time(self, mode="2D"):
        Ny = self.resolution_y
        Nz = self.nslices
        if mode == "2D":
            self.scan_time_2d = Ny * (self.nslices / self.current_2d_thickness_slices) * self.TR_2D * self.averages
        else:
            self.scan_time_3d =  Ny * Nz * self.TR_3D * self.averages
        
    def generate_volume(self):
        try:
            # Load the file
            img = nib.load('./datasets/t1_icbm_normal_1mm_pn3_rf20.mnc')
            self.base_volume = img.get_fdata()
            
            # --- NEW: Try to get the actual base thickness from header ---
            try:
                # .get_zooms() returns voxel sizes (e.g., [1.0, 1.0, 1.0])
                # Assuming the first dimension (shape[0]) corresponds to Z-direction here
                self.base_thickness_mm = img.header.get_zooms()[0]
            except:
                # Fallback if header is missing or weird
                self.base_thickness_mm = 1.0

            self.resolution_y = self.base_volume.shape[1]
            self.resolution_x = self.base_volume.shape[2]
            self.nslices = self.base_volume.shape[0]

            # Update slider max now that we know nslices
            if hasattr(self, 'slice_slider'):
                 self.slice_slider.setMaximum(self.nslices - 1)

            # Apply initial noise level
            self._apply_noise()

            self.kspace_2d = self.recon_2d = None
            self.kspace_3d = self.recon_3d = None
            print(f"Volume loaded. Shape: {self.base_volume.shape}, Base Thickness: {self.base_thickness_mm}mm")
            print(f"Signal range: {self.base_volume.min():.2f} to {self.base_volume.max():.2f}")

        except Exception as e:
             print(f"Error loading volume: {e}")
             # Fallback dummy volume if file missing
             self.base_volume = np.random.rand(181, 217, 181)
             self.nslices = 181
             self.resolution_y = 217
             self.resolution_x = 181
             self.base_thickness_mm = 1.0
             self._apply_noise()

    def _apply_noise(self):
        """Generates a noisy copy of the base volume based on the current noise level."""
        if self.base_volume is None:
            return
        
        # FIX: Scale noise relative to the signal range, not absolute percentage
        signal_range = self.base_volume.max() - self.base_volume.min()
        std_dev = (self.noise_level_percent / 100.0) * signal_range
        noise = np.random.normal(0, std_dev, self.base_volume.shape)
        self.noisy_volume = self.base_volume + noise
        
        print(f"Noise level: {self.noise_level_percent}%, std_dev: {std_dev:.4f}")

    def calculate_snr(self, image_slice, is_normalized=False):
        """
        Calculates SNR from a central tissue ROI and a corner air ROI.
        
        FIX: Added is_normalized flag to handle normalized vs raw data differently.
        For normalized data (0-1), we calculate SNR differently.
        """
        h, w = image_slice.shape
        # Central 20x20 ROI for tissue signal
        tissue_roi = image_slice[h//2-10:h//2+10, w//2-10:w//2+10]
        # Top-left 30x30 ROI for background noise
        air_roi = image_slice[0:30, 0:30]
        
        if is_normalized:
            # For normalized data, use contrast-to-noise ratio approach
            # SNR = (mean_signal - mean_background) / std_background
            signal_mean = np.mean(tissue_roi)
            background_mean = np.mean(air_roi)
            background_std = np.std(air_roi)
            snr = (signal_mean - background_mean) / (background_std + 1e-12)
        else:
            # For raw data: SNR = mean(signal) / std(noise)
            snr = np.mean(tissue_roi) / (np.std(air_roi) + 1e-12)
        
        return max(0, snr)  # Ensure non-negative SNR

    # -----------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------
    def on_generate(self):
        self.generate_volume()
        self.update_display()
        self.info.setText("Volume reloaded.")

    def on_slice_changed(self, val):
        self.slice_idx = int(val)
        self.slice_label.setText(f"Slice: {self.slice_idx}")
        self.update_display()

    def on_noise_changed(self, val):
        """Handler for the noise slider."""
        self.noise_level_percent = int(val)
        self.noise_label.setText(f"Added Noise: {self.noise_level_percent}%")
        self._apply_noise()
        # Clear previous reconstructions since noise changed
        self.kspace_2d = self.recon_2d = self.snr_2d = None
        self.kspace_3d = self.recon_3d = self.snr_3d = None
        self.update_display()

    # --- NEW: Handler for the thickness slider ---
    def on_thick_changed(self, val):
        self.current_2d_thickness_slices = int(val)
        self.thick_label.setText(f"Simulated 2D Thickness: {self.current_2d_thickness_slices} slices")
        
        # Calculate physical thickness in mm based on the base (native) thickness
        # If base is 1mm and we average 5 slices, physical thickness is 5mm.
        physical_thickness_mm = self.current_2d_thickness_slices * self.base_thickness_mm
        
        # Update comparison labels
        self.info_3d_thick.setText(f"3D Effective Thickness: {self.base_thickness_mm:.1f} mm")
        self.info_2d_thick.setText(f"2D Physical Thickness: {physical_thickness_mm:.1f} mm")
        # Update scan time / voxel info 
        self.update_info_labels()
        
    def on_fft2d(self):
        self.recon_3d = None
        self.kspace_3d = None
        self.snr_3d = None
        
        if self.noisy_volume is None:
            self.info.setText("No data")
            return
        
        # --- MODIFIED: Use dynamic slider value for averaging ---
        # Ensure we don't try to average past the end of the volume array
        end_slice = min(self.slice_idx + self.current_2d_thickness_slices, self.nslices)
        
        # If we are at the very last slice, just take that one slice to avoid empty array
        if self.slice_idx >= end_slice:
             data = self.noisy_volume[self.slice_idx, :, :]
        else:
             # Average the slices to simulate a thicker physical 2D acquisition
             data = np.mean(self.noisy_volume[self.slice_idx : end_slice, :, :], axis=0)
        
        # FIX: Calculate SNR on raw data BEFORE normalization
        snr_raw = self.calculate_snr(data, is_normalized=False)
        
        k = np.fft.fft2(data)
        k = np.fft.fftshift(k)

        # Inverse FFT
        recon = np.fft.ifft2(np.fft.ifftshift(k))
        recon = np.abs(recon)
        
        # Normalize for display
        recon_normalized = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

        self.kspace_2d = k
        self.recon_2d = recon_normalized
        self.snr_2d = snr_raw

        self.info.setText(f"2D FFT done. Simulated {self.current_2d_thickness_slices*self.base_thickness_mm:.1f}mm slice. SNR: {self.snr_2d:.1f}")
        self.update_display()

    def on_fft3d(self):
        self.recon_2d = None
        self.kspace_2d = None
        self.snr_2d = None
        
        if self.noisy_volume is None:
            self.info.setText("No data")
            return

        self.info.setText("Running 3D FFT... please wait...")
        QApplication.processEvents() # Keep UI responsive-ish

        k = np.fft.fftn(self.noisy_volume)
        k = np.fft.fftshift(k)
        
        recon = np.fft.ifftn(np.fft.ifftshift(k))
        recon = np.abs(recon)
        
        # FIX: Calculate SNR on raw reconstruction BEFORE normalization
        snr_raw = self.calculate_snr(recon[self.slice_idx], is_normalized=False)
        
        # Normalize for display
        recon_normalized = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

        self.kspace_3d = k
        self.recon_3d = recon_normalized
        self.snr_3d = snr_raw
        
        self.info.setText(f"3D FFT done. Effective slice thickness: {self.base_thickness_mm:.1f}mm. SNR: {self.snr_3d:.1f}")
        self.update_display()

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    def update_display(self):
        cmap = self.cmap_box.currentText()
        if self.noisy_volume is not None:
            # Show the current single slice of the *noisy* volume as the input reference
            img = self.base_volume[self.slice_idx]
            # Normalize for display
            img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-12)
            self.canvas_img.plot(img_normalized, title=f"Input Phantom (Slice {self.slice_idx})", cmap=cmap)
        
        # Update scan time / voxel size labels
        self.update_info_labels()
        
        if self.kspace_2d is not None:
            mag = np.log(1 + np.abs(self.kspace_2d))
            mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
            self.canvas_k.plot(mag, title="2D FFT k-space", cmap=cmap)
        elif self.kspace_3d is not None:
            mag = np.log(1 + np.abs(self.kspace_3d[self.slice_idx]))
            mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
            self.canvas_k.plot(mag, title=f"3D FFT k-space (Slice {self.slice_idx})", cmap=cmap)
        else:
            self.canvas_k.plot(np.zeros((self.resolution_y, self.resolution_x)), title="K-space", cmap=cmap)

        if self.recon_2d is not None:
            title = f"2D Recon ({self.current_2d_thickness_slices*self.base_thickness_mm:.0f}mm thick) | SNR: {self.snr_2d:.1f}"
            self.canvas_recon.plot(self.recon_2d, title=title, cmap=cmap)
        elif self.recon_3d is not None:
            # FIX: Recalculate SNR for the current slice being displayed
            # Use the stored raw reconstruction if available, otherwise use normalized
            snr_current = self.calculate_snr(self.recon_3d[self.slice_idx], is_normalized=True)
            title = f"3D Recon ({self.base_thickness_mm:.0f}mm thick) | SNR: {snr_current:.1f}"
            self.canvas_recon.plot(self.recon_3d[self.slice_idx], title=title, cmap=cmap)
        else:
            self.canvas_recon.plot(np.zeros((self.resolution_y, self.resolution_x)), title="Reconstruction", cmap=cmap)

    def update_info_labels(self):
        """Compute and update scan time and voxel size labels for both 2D and 3D modes."""
        try:
            # compute scan times
            self.compute_scan_time(mode="2D")
            self.compute_scan_time(mode="3D")

            # compute voxel sizes (vx, vy, vz) in mm
            vx2, vy2, vz2 = self.compute_voxel_size_mm(mode="2D")
            vx3, vy3, vz3 = self.compute_voxel_size_mm(mode="3D")

            # Format times (seconds) and voxel sizes
            t2 = getattr(self, 'scan_time_2d', None)
            t3 = getattr(self, 'scan_time_3d', None)
            t2m = f"{t2/60:.0f}m" if t2 is not None else "-"
            t3m = f"{t3/60:.0f}m" if t3 is not None else "-"

            self.scan_time_label.setText(f"Scan Time 2D/3D: {t2m} / {t3m}")
            self.voxel_size_label.setText(f"Voxel (2D): {vx2:.2f} x {vy2:.2f} x {vz2:.2f} mm  |  (3D): {vx3:.2f} x {vy3:.2f} x {vz3:.2f} mm")
        except Exception:
            # safe fallback
            self.scan_time_label.setText("Scan Time: -")
            self.voxel_size_label.setText("Voxel Size (mm): - x - x -")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FFTDemo()
    win.show()
    sys.exit(app.exec_())