# 3-DOF Robotic Arm Inverse & Forward Kinematics
'''
Now extended to 3-DOF:
 - theta0 : base rotation
 - theta1 : shoulder
 - theta2 : elbow
'''

import math
import numpy as np
import matplotlib.pyplot as plt

def inverse_calculation():
    # Link lengths
    L1 = 5
    L2 = 5
    
    # Target coordinates in 3D
    x, y, z = 3, 3, 4

    # Base rotation
    theta0 = np.arctan2(y, x)
    
    # Distance in base plane
    d = np.sqrt(x**2 + y**2)
    r = np.sqrt(d**2 + z**2)
    
    # Elbow angle
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2*L1*L2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)  # avoid domain errors
    theta2 = np.arccos(cos_theta2)
    
    # Shoulder angle
    alpha = np.arctan2(z, d)
    beta = np.arccos((L1**2 + r**2 - L2**2) / (2*L1*r))
    theta1 = alpha - beta
    
    # Convert to degrees
    print(f"Theta0 (base) = {np.degrees(theta0):.2f}°")
    print(f"Theta1 (shoulder) = {np.degrees(theta1):.2f}°")
    print(f"Theta2 (elbow) = {np.degrees(theta2):.2f}°")
    
    # Forward check
    x_fk = (L1*np.cos(theta1) + L2*np.cos(theta1+theta2)) * np.cos(theta0)
    y_fk = (L1*np.cos(theta1) + L2*np.cos(theta1+theta2)) * np.sin(theta0)
    z_fk = L1*np.sin(theta1) + L2*np.sin(theta1+theta2)
    
    print(f"Forward check: FK=({x_fk:.2f}, {y_fk:.2f}, {z_fk:.2f}) vs Target=({x}, {y}, {z})")
    
    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Joint positions
    x1 = L1*np.cos(theta1)*np.cos(theta0)
    y1 = L1*np.cos(theta1)*np.sin(theta0)
    z1 = L1*np.sin(theta1)
    
    x2 = x1 + L2*np.cos(theta1+theta2)*np.cos(theta0)
    y2 = y1 + L2*np.cos(theta1+theta2)*np.sin(theta0)
    z2 = z1 + L2*np.sin(theta1+theta2)
    
    ax.plot([0, x1, x2], [0, y1, y2], [0, z1, z2], '-o', label="Arm")
    ax.scatter(x, y, z, c="red", marker="x", s=100, label="Target")
    ax.scatter(0, 0, 0, c="black", s=80, label="Base")
    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3-DOF Robotic Arm IK")
    ax.legend()
    plt.show()


def forward_workspace():
    L1 = 4
    L2 = 3
    theta0_range = np.linspace(-180, 180, 60)
    theta1_range = np.linspace(0, 180, 30)
    theta2_range = np.linspace(0, 180, 30)

    X, Y, Z = [], [], []
    for t0 in theta0_range:
        for t1 in theta1_range:
            for t2 in theta2_range:
                th0 = np.radians(t0)
                th1 = np.radians(t1)
                th2 = np.radians(t2)
                
                x = (L1*np.cos(th1) + L2*np.cos(th1+th2)) * np.cos(th0)
                y = (L1*np.cos(th1) + L2*np.cos(th1+th2)) * np.sin(th0)
                z = L1*np.sin(th1) + L2*np.sin(th1+th2)
                
                X.append(x)
                Y.append(y)
                Z.append(z)

    # Plot workspace in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=1, alpha=0.3, c="blue")
    ax.scatter(0, 0, 0, c="red", s=50, label="Base")
    ax.set_title("3-DOF Arm Workspace")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


if __name__ == "__main__":
    inverse_calculation()
    forward_workspace()
