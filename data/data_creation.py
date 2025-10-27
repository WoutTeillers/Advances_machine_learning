import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


G = 1.0  # Gravitational constant
masses = [1.0, 1.0, 1.0]  # Masses of the three bodies

def three_body_equations(t, y):
    """
    We input a vector with x, y, x-velocity and y-velocity of the three
    bodies and return the derivative of this vector
    """
    
    
    dydt = np.zeros_like(y) # This is the derivative vector we return


    positions = np.array([[y[0], y[1]],
                          [y[4], y[5]],
                          [y[8], y[9]]])
    velocities = np.array([[y[2], y[3]],
                           [y[6], y[7]],
                           [y[10], y[11]]])

    # The derivatives of the place coordinates become velocities
    dydt[0] = velocities[0,0]
    dydt[1] = velocities[0,1]
    dydt[4] = velocities[1,0]
    dydt[5] = velocities[1,1]
    dydt[8] = velocities[2,0]
    dydt[9] = velocities[2,1]

    # The derivatives of the velocity coordinates become accelerations
    accelerations = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                accelerations[i] += G * masses[j] * r_vec / r_mag**3
                # This formula uses Newtons laws of gravitation

    dydt[2] = accelerations[0,0]
    dydt[3] = accelerations[0,1]
    dydt[6] = accelerations[1,0]
    dydt[7] = accelerations[1,1]
    dydt[10] = accelerations[2,0]
    dydt[11] = accelerations[2,1]

    return dydt


def get_trajectories():

    # Initial position and velocities as a 12-D vector
    y0 = np.array([
        -1.0, 0.0, 0.3471, 0.5327,   # The x, y, vx and vy of the first body
        1.0, 0.0, 0.3471, 0.5327,   # The x, y, vx and vy of the second body
        0.0, 0.0, -2*0.3471, -2*0.5327    # The x, y, vx and vy of the third body
    ])

    y0 = np.array([
        -1.0, 0.0, 0.1, 0.1 ,   # The x, y, vx and vy of the first body
        1.0, 0.0, 0.0, -0.1,   # The x, y, vx and vy of the second body
        0.0, 1.0, -0.1, 0.0    # The x, y, vx and vy of the third body
    ])

    # Time span for the simulation
    eval_time = 100
    steps = 1000 * eval_time
    t_span = (0, eval_time)
    t_eval = np.linspace(0, eval_time, steps)

    # Solve
    sol = solve_ivp(three_body_equations, t_span, y0, method='RK45', t_eval=t_eval)

    # time points
    t = sol.t  # shape (steps,)

    # reshape into (n_bodies, 4, n_times)
    n_bodies = 3
    state = sol.y.reshape(n_bodies, 4, -1)

    # positions and velocities
    x = state[:, 0, :]  # shape (3, steps)
    y = state[:, 1, :]  # shape (3, steps)
    vx = state[:, 2, :] # shape (3, steps)
    vy = state[:, 3, :] # shape (3, steps)

    plt.figure(figsize=(6, 6))
    for i in range(n_bodies):
        plt.plot(x[i], y[i], label=f"Body {i+1}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Three-body trajectories")
    plt.axis("equal")
    plt.legend()
    plt.show()

    return x,y,vx,vy,t

def plot_trajectories(x, x_pred=None, num_bodies=3):
    # x is (steps, 12) in order [x1,x2,x3, y1,y2,y3, vx1,vx2,vx3, vy1,vy2,vy3]
    colors = {0: 'r', 1: 'g', 2: 'b'}
    plt.figure(figsize=(6,6))
    for i in range(num_bodies):
     
        plt.plot(x[:, 4*i], x[:, 4*i+1], label=f"True Body {i+1}", linestyle='-')
        plt.plot(x_pred[:, 4*i], x_pred[:, 4*i+1], label=f"Pred Body {i+1}", linestyle='--')
        plt.scatter(x[0, 4*i], x[0, 4*i+1], color='blue', marker='o', s=50, edgecolor='black')
        plt.scatter(x_pred[:, 4*i], x_pred[:, 4*i+1], color='orange', s=15, edgecolor='black', alpha=0.6)
        
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Three-body trajectories")
    plt.axis("equal")
    plt.legend()
    plt.show()
