import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

fps = 30
# Parameters for the top
omega = 2                # Angular velocity
theta0 = np.radians(42.5)  # Initial tilt angle of the top
phi_dot = 1.5            # Precession rate
psi_dot = 2.5            # Spin rate

# Time array
t = np.linspace(0, 20, 100)

# Angular position functions
theta = theta0 * np.ones_like(t)
phi = phi_dot * t
psi = psi_dot * t

# Generate a cone mesh
n_points = 50
alpha = np.linspace(0, 2 * np.pi, n_points)
height = np.linspace(0, 1, n_points)
alpha, height = np.meshgrid(alpha, height)
cone_x = height * np.sin(theta0) * np.cos(alpha)
cone_y = height * np.sin(theta0) * np.sin(alpha)
cone_z = height * np.cos(theta0)

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel("x1'")
ax.set_ylabel("x2'")
ax.set_zlabel("x3'")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])

# Rotation function for body cone in space
def rotation_matrix(phi, theta, psi):
    c1, s1 = np.cos(phi), np.sin(phi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(psi), np.sin(psi)
    return np.array([
        [c1 * c3 - s1 * s2 * s3, -c1 * s3 - s1 * s2 * c3, s1 * c2],
        [s1 * c3 + c1 * s2 * s3, -s1 * s3 + c1 * s2 * c3, -c1 * c2],
        [c2 * s3, c2 * c3, s2]
    ])

# Animation update function
def update(i):
    ax.cla()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("x1'")
    ax.set_ylabel("x2'")
    ax.set_zlabel("x3'")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    
    # Plot space cone (stationary)
    ax.plot_surface(2*cone_x,2* cone_y, 3*cone_z, color='red', alpha=0.3, edgecolor='none', label="Space Cone")
    
    # Rotate body cone points
    R = rotation_matrix(phi[i], theta[i], psi[i])
    x_rot = R[0, 0] * 2*cone_x + R[0, 1] * 2*cone_y + R[0, 2] * cone_z
    y_rot = R[1, 0] * 2*cone_x + R[1, 1] * 2*cone_y + R[1, 2] * cone_z
    z_rot = R[2, 0] * 2*cone_x + R[2, 1] * 2*cone_y + R[2, 2] * cone_z
    
    # Plot rotating body cone
    ax.plot_surface(x_rot, y_rot, z_rot, color='blue', alpha=0.5, edgecolor='none', label="Body Cone")

# Create and display animation
ani = FuncAnimation(fig, update, frames=len(t), interval=1000/fps)
plt.show()
#%%
# Save the animation as a GIF
output_path = "C:/Users/Alex/Downloads/symmetric_top_motion3.gif"
ani.save(output_path, writer=PillowWriter(fps=fps))

print(f"Animation saved as {output_path}")

