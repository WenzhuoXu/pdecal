import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fenics import *
from tqdm import tqdm
import os

def solve_1d_burger(epsilon, length, n_x, dt, num_steps, expression_str=None, global_solution_idx=0):
    # Define mesh and function spaces
    mesh = IntervalMesh(n_x, 0, length)
    V = FunctionSpace(mesh, 'CG', 1)

    # Define initial condition
    x = SpatialCoordinate(mesh)
    u_init = Expression(expression_str, degree=2)
    u_n = interpolate(u_init, V)

    # Define test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)

    # Define weak form
    F = (u - u_n) / dt * v * dx + u_ * u.dx(0) * v * dx + epsilon * u.dx(0) * v.dx(0) * dx
    a, L = lhs(F), rhs(F)

    # Time-stepping
    t = 0

    # Prepare for visualization
    fig, ax = plt.subplots()
    x_vals = mesh.coordinates()
    ax.set_xlim(0, length)
    ax.set_ylim(0, 1)
    line, = ax.plot([], [], lw=2)

    # Initialize solution matrix
    u_val = []

    def update(frame, u_n, u, u_, t, dt):
        u_.assign(u_n)  # Update the non-linear term with the previous solution
        solve(a == L, u_)
        u_n.assign(u_)
        # u_val[frame, :] = u_n.compute_vertex_values()
        u_val.append(u_n.compute_vertex_values())
        t += dt
        line.set_data(x_vals, u_n.compute_vertex_values())
        return line,

    ani = animation.FuncAnimation(fig, update, frames=num_steps, fargs=(u_n, u, u_, t, dt),
                                interval=100, blit=True)
    plt.close()
    ani.save('./burger/vanishing_viscosity_{}.gif'.format(global_solution_idx), writer='pillow', fps=15)
    os.remove('./burger/vanishing_viscosity_{}.gif'.format(global_solution_idx))
    global_solution_idx += 1
    return np.array(u_val)


def solve_2d_burger(epsilon, length, n_x, dt, num_steps, expression_str=None, global_solution_idx=0):
    # Define mesh and function spaces
    mesh = RectangleMesh(Point(0, 0), Point(length, length), n_x, n_x)
    V = FunctionSpace(mesh, 'CG', 1)

    # Define initial condition
    x = SpatialCoordinate(mesh)
    u_init = Expression(expression_str, degree=2)
    u_n = interpolate(u_init, V)

    # Define test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)

    # Define weak form
    a = u * v * dx + epsilon * dot(grad(u), grad(v)) * dx
    L = u_init * v * dx

    # Time-stepping
    t = 0

    # Prepare for visualization
    fig, ax = plt.subplots()
    x_vals = mesh.coordinates()
    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    square, = ax.plot([], [], lw=2)

    # Initialize solution matrix
    u_val = []

    def update(frame, u_n, u, u_, t, dt):
        u_.assign(u_n)
        solve(a == L, u_, bcs=[DirichletBC(V, 0, "on_boundary")])
        u_n.assign(u_)
        # u_vals[frame, :, :] = u_n.compute_vertex_values().reshape((n_x+1, n_x+1))
        u_val.append(u_n.compute_vertex_values().reshape((n_x+1, n_x+1)))
        t += dt
        square.set_data(u_n.x, u_n.y)
        return square,

    ani = animation.FuncAnimation(fig, update, frames=num_steps, fargs=(u_n, u, u_, t, dt),
                                interval=100, blit=True)
    
    ani.save('./burger/vanishing_viscosity_{}.gif'.format(global_solution_idx), writer='pillow', fps=15)
    plt.close()
    global_solution_idx += 1
    return np.array(u_val)


def gen_random_expression_str():
    """
    generate a str expression for initial condition of burgers equation such as 'exp(-2*pow(x[0] - 1, 2))', 'sin(x[0])', etc.
    """
    function_type = np.random.choice(['exp', 'sin', 'cos', 'pow'])
    x_center = np.random.uniform(0, 3)
    if function_type == 'exp':
        return function_type + '(-2*pow(x[0] - ' + str(x_center) + ', 2))'
    elif function_type == 'pow':
        return function_type + '(x[0] - ' + str(x_center) + ', 2)'
    else:
        return function_type + '(x[0] - ' + str(x_center) + ')'
    

if __name__ == '__main__':
    # Parameters
    epsilon = 0.1
    length = 3.0
    n_x = 100
    dt = 0.01
    T = 2.0
    num_steps = int(T/dt)
    u_vals = []

    for i in tqdm(range(10000)):
        exp_ = gen_random_expression_str()
        u_val = solve_1d_burger(epsilon, length, n_x, dt, num_steps, exp_, i)
        u_vals.append(u_val)

    np.save('./burger/burger_results.npy', u_vals)