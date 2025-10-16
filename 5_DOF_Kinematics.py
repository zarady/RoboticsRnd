import numpy as np
import matplotlib.pyplot as plt

# If you want numerical IK using least_squares:
try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    print("scipy not available — IK will use simple random-search fallback (slower).")

# ---------- Helper transforms ----------
def Rz(theta):
    c = np.cos(theta); s = np.sin(theta)
    R = np.eye(4)
    R[0,0] = c; R[0,1] = -s
    R[1,0] = s; R[1,1] = c
    return R

def Ry(theta):
    c = np.cos(theta); s = np.sin(theta)
    R = np.eye(4)
    R[0,0] = c; R[0,2] = s
    R[2,0] = -s; R[2,2] = c
    return R

def Tx(L):
    T = np.eye(4)
    T[0,3] = L
    return T

# ---------- Forward kinematics (returns joint positions for plotting) ----------
def fk(angles, Ls):
    # angles: list/array [theta0, theta1, theta2, theta3, theta4] in radians
    theta0, theta1, theta2, theta3, theta4 = angles
    L1, L2, L3 = Ls

    # Build transforms
    T = np.eye(4)
    joints_world = [T.copy()]  # base frame

    # base yaw
    T = T @ Rz(theta0)
    # shoulder pitch then link1
    T = T @ Ry(theta1) @ Tx(L1)
    joints_world.append(T.copy())  # after link1

    # yaw at end of link1, then translate link2
    T = T @ Rz(theta2) @ Tx(L2)
    joints_world.append(T.copy())  # after link2

    # pitch then translate link3
    T = T @ Ry(theta3) @ Tx(L3)
    joints_world.append(T.copy())  # after link3 (wrist end BEFORE theta4)

    # final wrist pitch (no extra translation for position)
    T = T @ Ry(theta4)
    joints_world.append(T.copy())  # end-effector frame

    # Extract positions
    positions = [jw[0:3, 3].copy() for jw in joints_world]  # list of 3D points

    # end-effector position
    ee_pos = positions[-1]

    return ee_pos, positions

# ---------- Plot arm function ----------
def plot_arm(angles_deg, Ls):
    angles = np.radians(angles_deg)
    ee, positions = fk(angles, Ls)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=6)
    ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=80, label='End Effector')
    ax.scatter(0,0,0, c='k', s=60, label='Base')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3-Link Arm with two yaw joints (5 DOF)')
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.show()

# ---------- Numerical IK (least-squares) ----------
def ik_numerical(target_xyz, Ls, initial_guess_deg=None, joint_limits_deg=None, max_nfev=2000):
    """
    target_xyz: (x,y,z)
    Ls: (L1,L2,L3)
    initial_guess_deg: list of 5 angles in degrees
    joint_limits_deg: [(min,max),...] for each of 5 joints in degrees (optional)
    Returns angles in degrees (5,)
    """
    target = np.array(target_xyz, dtype=float)

    if initial_guess_deg is None:
        # reasonable default: all zeros
        initial_guess_deg = [0.0, 45.0, 0.0, 45.0, 0.0]

    x0 = np.radians(initial_guess_deg)

    # define residual: difference between fk position and target
    def residuals(x):
        ee, _ = fk(x, Ls)
        return (ee - target)

    if SCIPY_AVAILABLE:
        # handle bounds in radians if joint_limits provided
        if joint_limits_deg is not None:
            lb = np.array([lim[0] for lim in joint_limits_deg], dtype=float)
            ub = np.array([lim[1] for lim in joint_limits_deg], dtype=float)
            bounds = (np.radians(lb), np.radians(ub))
        else:
            bounds = (-np.inf, np.inf)

        res = least_squares(residuals, x0, bounds=bounds, max_nfev=max_nfev, xtol=1e-8)
        if not res.success:
            raise RuntimeError("IK solver did not converge: " + res.message)
        solution_rad = res.x
    else:
        # fallback: simple random-restart search (coarse)
        best = None
        best_err = 1e9
        rng = np.random.default_rng(0)
        for _ in range(2000):
            cand = x0 + np.radians(rng.uniform(-60,60,size=5))
            ee, _ = fk(cand, Ls)
            err = np.linalg.norm(ee - target)
            if err < best_err:
                best_err = err; best = cand.copy()
                if err < 1e-3:
                    break
        solution_rad = best

    return np.degrees(solution_rad)

# ---------- Example usage ----------
if __name__ == "__main__":
    # link lengths
    L1, L2, L3 = 4.0, 3.0, 2.0
    Ls = (L1, L2, L3)

    # Example target:
    target = (3.0, 3.0, 2.5)

    # Optional: joint limits in degrees (theta0..theta4)
    joint_limits = [(-180,180),   # theta0 yaw
                    (-90,90),     # theta1 pitch
                    (-180,180),   # theta2 yaw at link1 end
                    (-90,90),     # theta3 pitch
                    (-90,90)]     # theta4 pitch

    # initial guess (deg)
    initial_guess = [30, 20, 0, 20, 0]

    sol_deg = ik_numerical(target, Ls, initial_guess_deg=initial_guess, joint_limits_deg=joint_limits)
    print("IK solution (deg):", np.round(sol_deg,3))

    # Visualize
    plot_arm(sol_deg, Ls)

    # Verify FK result
    ee_pos, _ = fk(np.radians(sol_deg), Ls)
    print("FK at solution =>", np.round(ee_pos,4), " target =>", target)
