# 2-Link Robotic Arm Inverse Kinematics 
'''

Inverse Kineamtics 2-Link Robotic Arm
Learning purpose : to educate myself and appy what we have learn in university time

'''

import math
import numpy as np
import matplotlib.pyplot as plt

def inverse_calculation():
    # in a real world application, find the lenght of link1 and link2
    L1 = 5 # length of link1
    L2 = 5 # length of link2
    x,y = 2,4 # the coordinate of the end effector
    
    # talk about inverse, the purpose here is to find theta1 and theta2 of the robot
    
    '''
    inner thinking :
    sooo kita  guna basic triangle formula.
    find x = L1*cos(theta1) + L2*cos(theta1 + theta2)
    find y = L1*sin(theta1) + L2*sin(theta1 + theta2)  
    then kita simultaneous equation, cari satu satu thetanya...setel
    
    '''
    r_number = (x**2 + y**2)
    # print(r_number)
    r = (math.sqrt(r_number))
    #print(r)
    
    # Calculate theta2
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2*L1*L2)
    theta2 = np.arccos(cos_theta2)

    # Calculate theta1
    theta1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(theta2), L1 + L2*np.cos(theta2))

    # Convert to degrees
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)

    print(f"Theta1 = {theta1_deg:.2f} degrees")
    print(f"Theta2 = {theta2_deg:.2f} degrees")

    # Forward kinematics for visualization
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    # Plot arm
    plt.plot([0, x1, x2], [0, y1, y2], '-o')
    plt.plot(x, y, 'rx', label="Target")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def forward_calculation():
    
    # Link lengths
    L1 = 4
    L2 = 3

    # Joint limits (in degrees)
    theta1_range = np.linspace(0, 180, 100)  # 100 steps between 0 and 180
    theta2_range = np.linspace(0, 180, 100)

    # Store reachable points
    X, Y = [], []

    for t1 in theta1_range:
        for t2 in theta2_range:
            # Convert to radians
            th1 = np.radians(t1)
            th2 = np.radians(t2)
            
            # Forward kinematics
            x = L1*np.cos(th1) + L2*np.cos(th1 + th2)
            y = L1*np.sin(th1) + L2*np.sin(th1 + th2)
            
            X.append(x)
            Y.append(y)

    # Plot workspace
    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, s=2, c="blue", alpha=0.5, label="Reachable Points")
    
    # Add base point (robot center)
    plt.scatter(0, 0, c="red", s=80, marker="o", label="Robot Base")
    
    # Reference directions (0°, 90°, 180°, 270°)
    ref_len = L1 + L2 + 1  # make lines slightly longer than arm
    angles_deg = [0, 90, 180, 270]
    for angle in angles_deg:
        rad = np.radians(angle)
        x_ref = ref_len * np.cos(rad)
        y_ref = ref_len * np.sin(rad)
        plt.plot([0, x_ref], [0, y_ref], 'k--', alpha=0.5)  # dashed line
        plt.text(x_ref*1.05, y_ref*1.05, f"{angle}°", ha="center", va="center")
        
    plt.gca().set_aspect("equal")
    plt.title("2-Link Arm Workspace (Joint 0°–180°)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    inverse_calculation()
    forward_calculation()