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
        self.FOV_mm = 220.0            # Field of view (mm)
        
        # NEW: Base thickness of the loaded 3D volume (native resolution)
        self.base_thickness_mm = 1.0   
        # MODIFIED: This will now be dynamic based on the slider
        self.current_2d_thickness_slices = 5 

        self.TR = 5e-3                 # repetition time (s)
        self.noise_std = 0.01          # relative k-space noise level

        # Data placeholders
        self.volume = None
        self.kspace_2d = None
        self.recon_2d = None
        self.kspace_3d = None
        self.recon_3d = None
        self.snr_2d = None
        self.snr_3d = None
        self.scan_time_2d = None
        self.scan_time_3d = None

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

        # FFT buttons
        self.btn_fft2d = QPushButton("2D FFT (Simulated Thick Slice)")
        self.btn_fft2d.clicked.connect(self.on_fft2d)
        ctrl_layout.addWidget(self.btn_fft2d, 5, 0)
        self.btn_fft3d = QPushButton("3D FFT (Full Volume)")
        self.btn_fft3d.clicked.connect(self.on_fft3d)
        ctrl_layout.addWidget(self.btn_fft3d, 5, 1)

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
    def voxel_size_mm(self):
        # Simplified for this demo
        return 1.0, 1.0, self.base_thickness_mm

    def generate_volume(self):
        try:
            # Load the file
            img = nib.load('./datasets/t1_icbm_normal_1mm_pn3_rf20.mnc')
            self.volume = img.get_fdata()
            
            # --- NEW: Try to get the actual base thickness from header ---
            try:
                # .get_zooms() returns voxel sizes (e.g., [1.0, 1.0, 1.0])
                # Assuming the first dimension (shape[0]) corresponds to Z-direction here
                self.base_thickness_mm = img.header.get_zooms()[0]
            except:
                # Fallback if header is missing or weird
                self.base_thickness_mm = 1.0

            self.resolution_y = self.volume.shape[1]
            self.resolution_x = self.volume.shape[2]
            self.nslices = self.volume.shape[0]

            # Update slider max now that we know nslices
            if hasattr(self, 'slice_slider'):
                 self.slice_slider.setMaximum(self.nslices - 1)

            self.kspace_2d = self.recon_2d = None
            self.kspace_3d = self.recon_3d = None
            print(f"Volume loaded. Shape: {self.volume.shape}, Base Thickness: {self.base_thickness_mm}mm")

        except Exception as e:
             print(f"Error loading volume: {e}")
             # Fallback dummy volume if file missing
             self.volume = np.random.rand(181, 217, 181)
             self.nslices = 181
             self.resolution_y = 217
             self.resolution_x = 181
             self.base_thickness_mm = 1.0

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

    def on_fft2d(self):
        self.recon_3d = None
        self.kspace_3d = None
        
        if self.volume is None:
            self.info.setText("No data")
            return
        
        # --- MODIFIED: Use dynamic slider value for averaging ---
        # Ensure we don't try to average past the end of the volume array
        end_slice = min(self.slice_idx + self.current_2d_thickness_slices, self.nslices)
        
        # If we are at the very last slice, just take that one slice to avoid empty array
        if self.slice_idx >= end_slice:
             data = self.volume[self.slice_idx, :, :]
        else:
             # Average the slices to simulate a thicker physical 2D acquisition
             data = np.mean(self.volume[self.slice_idx : end_slice, :, :], axis=0)
        
        k = np.fft.fft2(data)
        k = np.fft.fftshift(k)

        # Inverse FFT
        recon = np.fft.ifft2(np.fft.ifftshift(k))
        recon = np.abs(recon)
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

        self.kspace_2d = k
        self.recon_2d = recon

        self.info.setText(f"2D FFT done. Simulated {self.current_2d_thickness_slices*self.base_thickness_mm:.1f}mm slice.")
        self.update_display()

    def on_fft3d(self):
        self.recon_2d = None
        self.kspace_2d = None
        
        if self.volume is None:
            self.info.setText("No data")
            return

        self.info.setText("Running 3D FFT... please wait...")
        QApplication.processEvents() # Keep UI responsive-ish

        k = np.fft.fftn(self.volume)
        k = np.fft.fftshift(k)
        
        recon = np.fft.ifftn(np.fft.ifftshift(k))
        recon = np.abs(recon)
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

        self.kspace_3d = k
        self.recon_3d = recon
        
        self.info.setText(f"3D FFT done. Effective slice thickness: {self.base_thickness_mm:.1f}mm")
        self.update_display()

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    def update_display(self):
        cmap = self.cmap_box.currentText()
        if self.volume is not None:
            # Show the current single slice as the "Ground Truth" reference
            img = self.volume[self.slice_idx]
            self.canvas_img.plot(img, title=f"Original Phantom (Slice {self.slice_idx})", cmap=cmap)

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
            self.canvas_recon.plot(self.recon_2d, title=f"2D Recon (Simulated {self.current_2d_thickness_slices*self.base_thickness_mm:.0f}mm thick)", cmap=cmap)
        elif self.recon_3d is not None:
            self.canvas_recon.plot(self.recon_3d[self.slice_idx], title=f"3D Recon (Effective {self.base_thickness_mm:.0f}mm thick)", cmap=cmap)
        else:
            self.canvas_recon.plot(np.zeros((self.resolution_y, self.resolution_x)), title="Reconstruction", cmap=cmap)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FFTDemo()
    win.show()
    sys.exit(app.exec_())