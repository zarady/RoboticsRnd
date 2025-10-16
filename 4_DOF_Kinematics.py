import numpy as np
import matplotlib.pyplot as plt

# ---------------- Forward Kinematics ----------------
def forward_kinematics(theta0, theta1, theta2, theta3, L1=4, L2=3, L3=2):
    # Convert to radians
    t0 = np.radians(theta0)
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)
    t3 = np.radians(theta3)

    # Effective planar projection
    x_plane = L1*np.cos(t1) + L2*np.cos(t1+t2) + L3*np.cos(t1+t2+t3)
    z = L1*np.sin(t1) + L2*np.sin(t1+t2) + L3*np.sin(t1+t2+t3)

    # Rotate around base yaw
    x = x_plane * np.cos(t0)
    y = x_plane * np.sin(t0)

    return x, y, z

# ---------------- Inverse Kinematics ----------------
def inverse_calculation(x, y, z, L1=4, L2=3, L3=2):
    # Step 1: base rotation
    theta0 = np.degrees(np.arctan2(y, x))

    # Step 2: distance in XY plane
    r = np.sqrt(x**2 + y**2)

    # Step 3: effective 2-link in XZ plane (combine L2+L3)
    L_eff = L2 + L3
    d = np.sqrt(r**2 + z**2)

    # Check reachability
    if d > (L1 + L_eff) or d < abs(L1 - L_eff):
        raise ValueError("Target point is outside reachable workspace!")

    # Law of cosines
    cos_theta2 = (d**2 - L1**2 - L_eff**2) / (2*L1*L_eff)
    theta2 = np.degrees(np.arccos(cos_theta2))

    # Shoulder angle
    theta1 = np.degrees(np.arctan2(z, r) - np.arctan2(L_eff*np.sin(np.radians(theta2)),
                                                      L1 + L_eff*np.cos(np.radians(theta2))))

    # Wrist fixed straight
    theta3 = 0

    return theta0, theta1, theta2, theta3

# ---------------- Plot Arm Configuration ----------------
def plot_arm(theta0, theta1, theta2, theta3, L1=4, L2=3, L3=2):
    # Convert to radians
    t0 = np.radians(theta0)
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)
    t3 = np.radians(theta3)

    # Joint positions
    x0, y0, z0 = 0, 0, 0  # base

    x1 = L1*np.cos(t1)*np.cos(t0)
    y1 = L1*np.cos(t1)*np.sin(t0)
    z1 = L1*np.sin(t1)

    x2 = x1 + L2*np.cos(t1+t2)*np.cos(t0)
    y2 = y1 + L2*np.cos(t1+t2)*np.sin(t0)
    z2 = z1 + L2*np.sin(t1+t2)

    x3 = x2 + L3*np.cos(t1+t2+t3)*np.cos(t0)
    y3 = y2 + L3*np.cos(t1+t2+t3)*np.sin(t0)
    z3 = z2 + L3*np.sin(t1+t2+t3)

    # Plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # Links
    ax.plot([x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3], '-o', c="blue", linewidth=2, markersize=6)
    ax.scatter(0,0,0, c="red", s=80, marker="o", label="Base")
    ax.scatter(x3,y3,z3, c="green", s=80, marker="x", label="End Effector")

    # Labels
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3-Link + Base Rotation Arm Configuration")
    ax.legend()
    plt.show()

# ---------------- Workspace Plot ----------------
def plot_workspace():
    L1, L2, L3 = 4, 3, 2
    theta0_range = np.linspace(0, 360, 30)  # base rotation
    theta1_range = np.linspace(0, 180, 15)  # shoulder
    theta2_range = np.linspace(0, 180, 15)  # elbow
    theta3_range = np.linspace(0, 180, 10)  # wrist

    X, Y, Z = [], [], []

    for t0 in theta0_range:
        for t1 in theta1_range:
            for t2 in theta2_range:
                for t3 in theta3_range:
                    x, y, z = forward_kinematics(t0, t1, t2, t3, L1, L2, L3)
                    X.append(x)
                    Y.append(y)
                    Z.append(z)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=1, c="blue", alpha=0.3)
    ax.scatter(0, 0, 0, c="red", s=80, marker="o", label="Robot Base")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3-Link + Base Rotation Workspace")
    plt.legend()
    plt.show()

# ---------------- Main ----------------
if __name__ == "__main__":
    # Example: Forward kinematics
    x, y, z = forward_kinematics(theta0=30, theta1=45, theta2=60, theta3=0)
    print(f"FK → End Effector: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    # Example: Inverse kinematics for target
    target = (3,3,4)
    t0, t1, t2, t3 = inverse_calculation(*target)
    print(f"IK → For target {target}: θ0={t0:.2f}, θ1={t1:.2f}, θ2={t2:.2f}, θ3={t3:.2f}")

    # Plot arm at IK solution
    plot_arm(t0, t1, t2, t3)

    # Plot workspace
    plot_workspace()
