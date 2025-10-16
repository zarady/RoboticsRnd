import numpy as np

def dh_matrix(theta, d, a, alpha):
    """Return the DH transformation matrix."""
    ct = np.cos(np.radians(theta))
    st = np.sin(np.radians(theta))
    ca = np.cos(np.radians(alpha))
    sa = np.sin(np.radians(alpha))

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,      ca,       d],
        [0,   0,       0,        1]
    ])

def forward_kinematics(theta_list, d_list, a_list, alpha_list):
    """Compute the final transformation matrix using DH parameters."""
    T = np.eye(4)
    for i in range(len(theta_list)):
        Ti = dh_matrix(theta_list[i], d_list[i], a_list[i], alpha_list[i])
        print(f"T{i+1} = \n{Ti}\n")  # Show step by step
        T = np.dot(T, Ti)  # Multiply chain
    return T

if __name__ == "__main__":
    # Example: 4 DOF arm
    theta = [90, 0, 0, 0]    # joint angles (deg)
    d = [0, 0, 3, 0]         # offset along z
    a = [0, 3, 3, 4]         # link lengths along x
    alpha = [0, 0, 90, 0]    # twist angles
    
    T_final = forward_kinematics(theta, d, a, alpha)
    
    # Extract end-effector position (x,y,z)
    end_effector_pos = T_final[0:3, 3]
    print("End effector position (x, y, z):", end_effector_pos)
    print("Final Transformation Matrix (End Effector Pose):\n", T_final)
