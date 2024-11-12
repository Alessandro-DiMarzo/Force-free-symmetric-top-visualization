import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#%%
# Parameters (adjust these as needed)
A = 1.0  # Amplitude of oscillation
I3 = 1
I1 = 25
Omega = (I3-I1)/I1  # Angular velocity for precession
t_max = 10  # Duration of the animation in seconds
fps = 30  # Frames per second

# Generate time points
t = np.linspace(0, t_max, t_max * fps)

# Angular velocities from the solution
omega_1 = A * np.cos(Omega * t)
omega_2 = A * np.sin(Omega * t)
omega_3 = np.ones_like(t)  # Constant angular velocity along the z-axis

# Set up the figure and axis
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
plt.tick_params(left = False, bottom = False) 
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
# Initial orientation of the top
top_line, = ax.plot([], [], [], 'b-', lw=2)
top_point, = ax.plot([], [], [], 'ro')

# Update function for the animation
def update(frame):
    # Calculate current angular velocity components
    wx = omega_1[frame]
    wy = omega_2[frame]
    wz = omega_3[frame]
    
    # Top orientation vector
    orientation_vector = np.array([wx, wy, wz]) / np.linalg.norm([wx, wy, wz])

    # Update the line representing the top
    top_line.set_data([0, orientation_vector[0]], [0, orientation_vector[1]])
    top_line.set_3d_properties([0, orientation_vector[2]])

    # Update the red point representing the tip of the top
    top_point.set_data(orientation_vector[0], orientation_vector[1])
    top_point.set_3d_properties(orientation_vector[2])

    return top_line, top_point

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=1000/fps)

# Display the animation
plt.show()

# Save the animation as a GIF
output_path = "C:/Users/Alex/Downloads/symmetric_top_motion1.gif"
ani.save(output_path, writer=PillowWriter(fps=fps))

print(f"Animation saved as {output_path}")

