import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt




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
    G = 1.0  # Gravitational constant
    masses = [1.0, 1.0, 1.0]  # Masses of the three bodies
    # Initial position and velocities as a 12-D vector
    y0 = np.array([
        -1.0, 0.0, 0.3471, 0.5327,   # The x, y, vx and vy of the first body
        1.0, 0.0, 0.3471, 0.5327,   # The x, y, vx and vy of the second body
        0.0, 0.0, -2*0.3471, -2*0.5327    # The x, y, vx and vy of the third body
    ])
    # I took these randomly but these initial conditions resulted in a nice plot.

    eval_time = 10
    steps = 10000
    t_span = (0, eval_time)
    t_eval = np.linspace(0, eval_time, steps)

    # Solve
    sol = solve_ivp(three_body_equations, t_span, y0, method='RK45', t_eval=t_eval)

    # Plotting to see whether our initial conditions result in chaotic behaviour
    x1, y1 = sol.y[0], sol.y[1]    # The position of body 1
    x2, y2 = sol.y[4], sol.y[5]    # The position of body 2
    x3, y3 = sol.y[8], sol.y[9]    # The position of body 3
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

    # # Plotting to see whether our initial conditions result in chaotic behaviour
    # x1, y1 = sol.y[0], sol.y[1]    # The position of body 1
    # x2, y2 = sol.y[4], sol.y[5]    # The position of body 2
    # x3, y3 = sol.y[8], sol.y[9]    # The position of body 3

    plt.figure(figsize=(6, 6))
    for i in range(n_bodies):
        plt.plot(x[i], y[i], label=f"Body {i+1}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Three-body trajectories")
    plt.axis("equal")
    plt.legend()
    plt.show()
    print(type(sol))
    return x,y,vx,vy,t