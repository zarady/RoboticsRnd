import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QLineEdit, QPushButton, QGridLayout, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------- Forward Kinematics ----------------
def forward_kinematics(theta0, theta1, theta2, theta3, L1=4, L2=3, L3=2):
    t0, t1, t2, t3 = map(np.radians, [theta0, theta1, theta2, theta3])
    x1 = L1*np.cos(t1)*np.cos(t0)
    y1 = L1*np.cos(t1)*np.sin(t0)
    z1 = L1*np.sin(t1)

    x2 = x1 + L2*np.cos(t1+t2)*np.cos(t0)
    y2 = y1 + L2*np.cos(t1+t2)*np.sin(t0)
    z2 = z1 + L2*np.sin(t1+t2)

    x3 = x2 + L3*np.cos(t1+t2+t3)*np.cos(t0)
    y3 = y2 + L3*np.cos(t1+t2+t3)*np.sin(t0)
    z3 = z2 + L3*np.sin(t1+t2+t3)

    return [(0,0,0), (x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]


# ---------------- GUI Class ----------------
class RobotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4 DOF Robot Arm GUI")
        self.setGeometry(100, 100, 1200, 700)

        # Default angles
        self.angles = [30, 45, 60, 0]  # theta0..theta3

        # Layout: left (plot) + right (controls)
        layout = QHBoxLayout(self)

        # Left: 3D Plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 2)

        # Right: Controls
        control_layout = QVBoxLayout()
        layout.addLayout(control_layout, 1)

        # Title + Description
        title = QLabel("4 DOF Robot")
        title.setStyleSheet("font-size: 30px; font-weight: bold;")
        control_layout.addWidget(title, alignment=Qt.AlignCenter)
        
        # Reduce spacing between title and desc
        control_layout.setSpacing(2)   # smaller gap

        desc = QLabel(
            "This robot consists of 3 links with a rotational base.\n"
            "It has 4 Degrees of Freedom (DOF):\n\n"
            " ◦ Base rotation   ◦\n"
            " ◦ Shoulder joint ◦\n"
            " ◦ Elbow joint     ◦\n"
            " ◦ Wrist joint      ◦\n\n"
            "The system can move the end effector in 3D space, and you can\n"
            "control it either by adjusting the joint angles or directly setting\n"
            "the end effector position (x, y, z).\n\n\n\n\n"
        )
        desc.setWordWrap(True)
        control_layout.addWidget(desc, alignment=Qt.AlignCenter)

        # End Effector Position (editable)
        ee_group = QGroupBox("End Effector Position (x, y, z)")
        ee_layout = QHBoxLayout()
        self.ee_boxes = []
        for i, default in enumerate([3, 3, 4]):
            box = QLineEdit(str(default))
            box.setFixedWidth(60)
            box.editingFinished.connect(self.update_from_ee)
            ee_layout.addWidget(QLabel("xyz"[i] + ":"))
            ee_layout.addWidget(box)
            self.ee_boxes.append(box)
        ee_group.setLayout(ee_layout)
        control_layout.addWidget(ee_group)

        # Joint sliders and textboxes
        self.sliders = []
        self.angle_boxes = []
        joint_group = QGroupBox("Joint Angles (degrees)")
        joint_layout = QGridLayout()
        labels = ["Base θ0", "Shoulder θ1", "Elbow θ2", "Wrist θ3"]

        for i, label in enumerate(labels):
            joint_layout.addWidget(QLabel(label), i, 0)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 180 if i>0 else 360)
            slider.setValue(self.angles[i])
            slider.valueChanged.connect(self.update_from_sliders)
            joint_layout.addWidget(slider, i, 1)
            self.sliders.append(slider)

            box = QLineEdit(str(self.angles[i]))
            box.setFixedWidth(60)
            box.editingFinished.connect(self.update_from_textboxes)
            joint_layout.addWidget(box, i, 2)
            self.angle_boxes.append(box)

        joint_group.setLayout(joint_layout)
        control_layout.addWidget(joint_group)

        # Exit Button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        exit_button.setFixedWidth(80)
        control_layout.addWidget(exit_button, alignment=Qt.AlignRight)

        # Initial plot
        self.update_plot()

    # ---------------- Update Functions ----------------
    def update_plot(self):
        pts = forward_kinematics(*self.angles)
        xs, ys, zs = zip(*pts)

        # Clear and redraw
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=6, c="blue")
        ax.scatter(0,0,0, c="red", s=80, marker="o", label="Base")
        ax.scatter(xs[-1], ys[-1], zs[-1], c="green", s=80, marker="x", label="End Effector")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("4 DOF Robot Arm")
        ax.legend()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-2, 10)

        self.canvas.draw()

        # Update end effector textboxes
        x, y, z = xs[-1], ys[-1], zs[-1]
        for box, val in zip(self.ee_boxes, [x,y,z]):
            box.setText(f"{val:.2f}")

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
            self.angles[i] = max(0, min(360 if i==0 else 180, val))
            self.sliders[i].setValue(int(self.angles[i]))
        self.update_plot()

    def update_from_ee(self):
        # Placeholder: full IK not added here
        # For now, just refresh plot
        self.update_plot()


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotApp()
    window.show()
    sys.exit(app.exec_())
