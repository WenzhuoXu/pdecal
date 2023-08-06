import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fenicstools.Interpolation import interpolate_nonmatching_mesh
from fenics import *
from tqdm import tqdm
import h5py
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


def generate_2d_mesh(length, n_x):
    mesh = RectangleMesh(Point(0, 0), Point(length, length), n_x, n_x)
    # save mesh to h5 file
    with h5py.File('./burger/mesh_{}.h5'.format(n_x), 'w') as f:
        X = mesh.coordinates()
            # X = [points[:, i] for i in range(2)]
        edges(mesh)
            # define connectivity in COO format
        lines = np.zeros((2 * mesh.num_edges(), 2), dtype=np.int32)
        line_lengths = np.zeros(2 * mesh.num_edges(), dtype=np.int32)

        for i, edge in enumerate(edges(mesh)):
            lines[2*i, :] = edge.entities(0)
            lines[2*i+1, :] = np.flipud(edge.entities(0))
            line_lengths[2*i] = edge.length()
            line_lengths[2*i+1] = edge.length()

        f.create_dataset("X", data=X)
        f.create_dataset("lines", data=lines)
        f.create_dataset("line_lengths", data=line_lengths)
    return mesh

def solve_2d_burger(mesh, mesh_high, epsilon, dt, num_steps, expression_str=None, global_solution_idx=0):
    # Define mesh and function spaces
    # mesh = RectangleMesh(Point(0, 0), Point(length, length), n_x, n_x)
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = VectorFunctionSpace(mesh_high, 'P', 2)

    # Define initial condition
    x = SpatialCoordinate(mesh)
    u_init = Expression((expression_str, expression_str), degree=2)
    u_n = interpolate(u_init, V)

    # Define test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant((0, 0))
    u_ = Function(V)

    # Define weak form
    F = inner((u - u_n) / dt, v)*dx - dot(v, dot(u_n, grad(u)))*dx \
        + epsilon*inner(grad(u), grad(v))*dx
    a = lhs(F)
    L = rhs(F)

    # Time-stepping
    t = 0

    # Prepare for visualization
    u_vector = u_n.compute_vertex_values(mesh)
    # compute the magnitude of the velocity
    # u_mag = np.sqrt(u_vector[:len(u_vector)//2]**2 + u_vector[len(u_vector)//2:]**2)

    # cont = [ax.contourf(x_vals[:, 0].reshape((n_x+1, n_x+1)), x_vals[:, 1].reshape((n_x+1, n_x+1)), u_mag.reshape((n_x+1, n_x+1)))]

    # Initialize solution matrix
    u_val = []

    def update(u_n, u_, t, dt):
        # global cont
        u_.assign(u_n)
        solve(a == L, u_)
        u_n.assign(u_)
        # u_vals[frame, :, :] = u_n.compute_vertex_values().reshape((n_x+1, n_x+1))
        # interpolate u_n to high resolution mesh
        u_n_high = interpolate_nonmatching_mesh(u_n, Q)
        u_vector = u_n_high.compute_vertex_values(mesh_high)
        # compute the magnitude of the velocity
        u_mag = np.sqrt(u_vector[:len(u_vector)//2]**2 + u_vector[len(u_vector)//2:]**2)
        u_val.append(u_mag)

        t += dt
        
    for i in range(num_steps):
        update(u_n, u_, t, dt)
    
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
    

def gen_random_expression_str_2d():
    """
    generate a str expression for initial condition of burgers equation using a Gaussian initial velocity distribution. The center of the Gaussian is randomly generated.
    """
    x_center = np.random.uniform(0, 3)
    y_center = np.random.uniform(0, 3)
    return 'exp(-2*pow(x[0] - ' + str(x_center) + ', 2) - 2*pow(x[1] - ' + str(y_center) + ', 2))'
    

if __name__ == '__main__':
    # Parameters
    epsilon = 0.01
    length = 3.0
    n_x = 100
    dt = 0.1
    T = 10.0
    num_steps = int(T/dt)
    # u_vals = []
    mesh_resolutions = [10, 20, 40, 80]
    mesh_all = [generate_2d_mesh(length, n_x) for n_x in mesh_resolutions]
    u_val_res_1 = []
    u_val_res_2 = []
    u_val_res_3 = []
    u_val_res_4 = []

    for i in tqdm(range(800)):
        exp_ = gen_random_expression_str_2d()
        for i in range(len(mesh_resolutions)):
            u_val = solve_2d_burger(mesh_all[i], mesh_all[3], epsilon, dt, num_steps, exp_, i)
            if i == 0:
                u_val_res_1.append(u_val)
            elif i == 1:
                u_val_res_2.append(u_val)
            elif i == 2:
                u_val_res_3.append(u_val)
            elif i == 3:
                u_val_res_4.append(u_val)

    
    with h5py.File('solution_{}.h5'.format(mesh_resolutions[0]), 'w') as f:
        for i in range(800):
            f.create_group('{}'.format(i))
            f['{}'.format(i)].create_dataset('u', data=u_val_res_1[i])

    with h5py.File('solution_{}.h5'.format(mesh_resolutions[1]), 'w') as f:
        for i in range(800):
            f.create_group('{}'.format(i))
            f['{}'.format(i)].create_dataset('u', data=u_val_res_2[i])

    with h5py.File('solution_{}.h5'.format(mesh_resolutions[2]), 'w') as f:
        for i in range(800):
            f.create_group('{}'.format(i))
            f['{}'.format(i)].create_dataset('u', data=u_val_res_3[i])

    with h5py.File('solution_{}.h5'.format(mesh_resolutions[3]), 'w') as f:
        for i in range(800):
            f.create_group('{}'.format(i))
            f['{}'.format(i)].create_dataset('u', data=u_val_res_4[i])

