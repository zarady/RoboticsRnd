import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QLineEdit, QPushButton, QGridLayout, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------- DH PARAMETERS ----------------
# Interpreting your table:
# Row 0 = base -> first joint (fixed transform)
# Rows 1..6 = 6 actuated joints (d, alpha, a, theta_range_min-theta_range_max)
# Format per row: (d, alpha_deg, a, (theta_min_deg, theta_max_deg))
# alpha_deg is indicator for the next  link, means when it said 90 degree, it shows the degree for the next link. Thank you
DH = [
    (0, 0, 0, (0, 270)),    # row 0: base offset (fixed, theta ignored)
    (2, 90, 0, (0, 270)),    # joint 1
    (0, -90, 10, (0, 180)),   # joint 2
    (2, 90, 0, (0, 270)),   # joint 3
    (8, -90, 0, (90, 270)),    # joint 4
    (-2, 90, 0,(0, 360)),    # joint 5
    (2, 0, 0, (0, 270)),    # joint 6
]

# Note: We will let joints correspond to DH[1]..DH[6] (6 DOF).
# DH[0] is applied first as a fixed transform (theta = 0).

# ---------------- DH helpers ----------------
def dh_transform(d, alpha_deg, a, theta_deg):
    """Return 4x4 homogeneous transform using standard DH (degrees)."""
    a = float(a)
    d = float(d)
    alpha = np.radians(alpha_deg)
    theta = np.radians(theta_deg)
    ca = np.cos(alpha); sa = np.sin(alpha)
    ct = np.cos(theta); st = np.sin(theta)
    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,    sa,     ca,    d],
        [0.,   0.,     0.,    1.]
    ], dtype=float)
    return T

def forward_kinematics_from_thetas(thetas):
    """
    thetas: list of 6 joint angles in degrees for joints 1..6.
    Returns list of joint positions [(x,y,z) ...] including base (origin).
    """
    if len(thetas) != 6:
        raise ValueError("Expect 6 joint angles.")
    # Start from base frame at origin
    positions = [(0.0, 0.0, 0.0)]
    # Apply fixed base transform (DH[0]) with theta=0
    T = dh_transform(DH[0][0], DH[0][1], DH[0][2], 0.0)
    # Then for each actuated joint i (1..6), apply its DH transform using theta from thetas
    for i, theta in enumerate(thetas, start=1):
        d, alpha, a, _ = DH[i]
        Ti = dh_transform(d, alpha, a, theta)
        T = T @ Ti
        pos = T[:3, 3]
        positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
    return positions  # length 7: base + 6 joint positions (end effector is last)

# ---------------- GUI Class ----------------
class RobotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6 DOF Robot Arm (DH-based) GUI")
        self.setGeometry(100, 100, 1300, 760)

        # Default angles (6 joints). Choose mid-range defaults.
        self.angles = []
        for i in range(1, 7):
            rng = DH[i][3]
            mid = (rng[0] + rng[1]) / 2.0
            self.angles.append(int(mid))

        # Layout: left (plot) + right (controls)
        layout = QHBoxLayout(self)

        # Left: 3D Plot
        self.figure = Figure(figsize=(6,6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 2)

        # Right: Controls
        control_layout = QVBoxLayout()
        layout.addLayout(control_layout, 1)

        # Title + Description
        title = QLabel("6 DOF Robot (DH parameters)")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        control_layout.addWidget(title, alignment=Qt.AlignCenter)
        control_layout.setSpacing(6)

        desc = QLabel(
            "This robot uses the supplied Denavit–Hartenberg parameters.\n\n"
            "You can control the six joint angles using sliders or text boxes.\n"
            "End effector position is shown and updated from forward kinematics.\n\n"
            "Note: Inverse kinematics is not implemented here (EE boxes are placeholders)."
        )
        desc.setWordWrap(True)
        control_layout.addWidget(desc, alignment=Qt.AlignCenter)

        # End Effector Position (editable)
        ee_group = QGroupBox("End Effector Position (x, y, z)")
        ee_layout = QHBoxLayout()
        self.ee_boxes = []
        for i, default in enumerate([0.0, 0.0, 0.0]):
            box = QLineEdit(str(default))
            box.setFixedWidth(80)
            box.editingFinished.connect(self.update_from_ee)
            ee_layout.addWidget(QLabel("xyz"[i] + ":"))
            ee_layout.addWidget(box)
            self.ee_boxes.append(box)
        ee_group.setLayout(ee_layout)
        control_layout.addWidget(ee_group)

        # Joint sliders and textboxes
        self.sliders = []
        self.angle_boxes = []
        joint_group = QGroupBox("Joint Angles (degrees) - joints 1..6")
        joint_layout = QGridLayout()
        labels = [f"Joint {i} θ{i}" for i in range(1,7)]

        for i, label in enumerate(labels):
            joint_layout.addWidget(QLabel(label), i, 0)

            slider = QSlider(Qt.Horizontal)
            j_idx = i + 1  # DH index
            rng = DH[j_idx][3]
            min_r, max_r = int(rng[0]), int(rng[1])
            # Allow full 360 if max>360 (but not needed here)
            slider.setRange(min_r, max_r)
            slider.setValue(self.angles[i])
            slider.valueChanged.connect(self.update_from_sliders)
            slider.setSingleStep(1)
            joint_layout.addWidget(slider, i, 1)
            self.sliders.append(slider)

            box = QLineEdit(str(self.angles[i]))
            box.setFixedWidth(80)
            box.editingFinished.connect(self.update_from_textboxes)
            joint_layout.addWidget(box, i, 2)
            self.angle_boxes.append(box)

            # Add a small label for (a,d,alpha) for reference
            d, alpha, a, _ = DH[j_idx]
            joint_layout.addWidget(QLabel(f"a={a}, d={d}, α={alpha}°"), i, 3)

        joint_group.setLayout(joint_layout)
        control_layout.addWidget(joint_group)

        # Quick preset buttons (optional useful features)
        preset_layout = QHBoxLayout()
        home_btn = QPushButton("Home (mid)")
        home_btn.clicked.connect(self.set_home)
        preset_layout.addWidget(home_btn)

        zero_btn = QPushButton("Zero")
        zero_btn.clicked.connect(self.set_zero)
        preset_layout.addWidget(zero_btn)

        control_layout.addLayout(preset_layout)

        # Exit Button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        exit_button.setFixedWidth(100)
        control_layout.addWidget(exit_button, alignment=Qt.AlignRight)

        # Initial plot
        self.update_plot()

    # ---------------- Update Functions ----------------
    def update_plot(self):
        pts = forward_kinematics_from_thetas(self.angles)
        xs, ys, zs = zip(*pts)

        # Clear and redraw
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=6)
        ax.scatter(0,0,0, s=80, marker="o", label="Base")
        ax.scatter(xs[-1], ys[-1], zs[-1], s=80, marker="x", label="End Effector")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("6 DOF Robot Arm (FK)")
        ax.legend()
        # Auto-limits based on link lengths (simple heuristic)
        all_a = sum(abs(row[2]) for row in DH)
        lim = max(all_a, 10)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim/2, lim)

        self.canvas.draw()

        # Update end effector textboxes
        x, y, z = xs[-1], ys[-1], zs[-1]
        for box, val in zip(self.ee_boxes, [x,y,z]):
            box.setText(f"{val:.3f}")

    def update_from_sliders(self):
        for i, slider in enumerate(self.sliders):
            self.angles[i] = slider.value()
            self.angle_boxes[i].setText(str(self.angles[i]))
        self.update_plot()

    def update_from_textboxes(self):
        for i, box in enumerate(self.angle_boxes):
            try:
                val = float(box.text())
            except ValueError:
                val = self.sliders[i].value()
            # clamp to slider range
            min_r = self.sliders[i].minimum()
            max_r = self.sliders[i].maximum()
            val_clamped = max(min_r, min(max_r, val))
            self.angles[i] = val_clamped
            self.sliders[i].setValue(int(val_clamped))
            self.angle_boxes[i].setText(str(int(val_clamped)))
        self.update_plot()

    def update_from_ee(self):
        # Placeholder: full IK not added here
        # Show a friendly message that IK is not implemented
        QMessageBox.information(self, "Not Implemented", "Inverse kinematics not implemented. End effector boxes are read-only placeholders for FK result.")
        # Keep current FK plot
        self.update_plot()

    # Presets
    def set_home(self):
        for i in range(6):
            rng = DH[i+1][3]
            mid = int((rng[0] + rng[1]) / 2.0)
            self.angles[i] = mid
            self.sliders[i].setValue(mid)
            self.angle_boxes[i].setText(str(mid))
        self.update_plot()

    def set_zero(self):
        for i in range(6):
            min_r = self.sliders[i].minimum()
            # choose 0 if inside range else min
            val = 0 if (min_r <= 0 <= self.sliders[i].maximum()) else min_r
            self.angles[i] = val
            self.sliders[i].setValue(int(val))
            self.angle_boxes[i].setText(str(int(val)))
        self.update_plot()

# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotApp()
    window.show()
    sys.exit(app.exec_())
