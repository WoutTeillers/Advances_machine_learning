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

    # plt.figure(figsize=(6, 6))
    # for i in range(n_bodies):
    #     plt.plot(x[i], y[i], label=f"Body {i+1}")

    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Three-body trajectories")
    # plt.axis("equal")
    # plt.legend()
    # plt.show()

    return x,y,vx,vy,t

def plot_trajectories(x, x_pred, num_bodies=3):
    """
    Plot 2D trajectories of multiple bodies comparing true and predicted positions.

    This function assumes that the state vector for each body is organized as
    [x, y, vx, vy, ...] in `x` and `x_pred`. It plots the x-y trajectories 
    of `num_bodies` bodies with solid lines for true positions and dashed lines
    for predicted positions.

    Args:
        x (np.ndarray): True trajectories with shape
            (num_time_steps, num_features). The features should be ordered per
            body as [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...].
        x_pred (np.ndarray): Predicted trajectories, same shape
            and feature ordering as `x`.
        num_bodies (int, optional): Number of bodies to plot. Defaults to 3.

    Raises:
        TypeError: If `x` or `x_pred` is not a NumPy array.

    Returns:
        None. Displays a Matplotlib figure with the trajectories.
    
    Notes:
        - Only the x and y positions (first two features per body) are plotted.
        - Velocities (vx, vy) are ignored for plotting.
        - Line styles: '-' for true trajectory, '--' for predicted trajectory.
    """
    # x is (steps, 12) in order [x1,x2,x3, y1,y2,y3, vx1,vx2,vx3, vy1,vy2,vy3]
    colors = {0: 'r', 1: 'g', 2: 'b'}
    plt.figure(figsize=(6,6))
    for i in range(num_bodies):
    
        plt.plot(x[:, i], x[:, i+3], label=f"True Body {i+1}", linestyle='-')
        plt.plot(x_pred[:, i], x_pred[:, i+3], label=f"Pred Body {i+1}", linestyle='--')
        plt.scatter(x[0, i], x[0, i+3], color='blue', marker='o', s=50, edgecolor='black')
        plt.scatter(x_pred[:, i], x_pred[:, i+3], color='orange', s=15, edgecolor='black', alpha=0.6)
        
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Three-body trajectories")
    plt.axis("equal")
    plt.legend()
    plt.show()


def plot_boxplots(data):
    data = np.asarray(data)
    labels = [
        "x1", "x2", "x3",
        "y1", "y2", "y3",
        "vx1", "vx2", "vx3",
        "vy1", "vy2", "vy3"
    ]

    plt.figure(figsize=(10,6))
    plt.boxplot(data, labels=labels)
    plt.title("Distribution of Positions and Velocities")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()

def plot_velocity_magnitude(data):
    data = np.asarray(data)
    vxs = data[:, [2, 6, 10]]
    vys = data[:, [3, 7, 11]]
    vmag = np.sqrt(vxs**2 + vys**2)  # (n, 3)

    plt.figure(figsize=(8,5))
    for i in range(3):
        plt.plot(vmag[:, i], label=f"Body {i+1}")
    plt.xlabel("Timestep")
    plt.ylabel("Velocity magnitude")
    plt.title("Velocity over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def transform_data(data, window_size=10, test_size=0.2, forecast_horizon=10):
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    print(data.shape)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(data)

    # Build sliding windows with forecast horizon
    X = np.array([scaled[i:i+window_size] for i in range(len(scaled) - window_size - forecast_horizon)])
    y = np.array([scaled[i+window_size+forecast_horizon-1] for i in range(len(scaled) - window_size - forecast_horizon)])

    # Split train/test
    train_size = int((1 - test_size) * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler
    