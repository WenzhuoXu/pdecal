import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fenics import *

# Parameters
epsilon = 0.1
length = 3.0
n_x = 100
dt = 0.01
T = 2.0
num_steps = int(T/dt)

def solve_1d_burger(epsilon, length, n_x, dt, num_steps):
    # Define mesh and function spaces
    mesh = IntervalMesh(n_x, 0, length)
    V = FunctionSpace(mesh, 'CG', 1)

    # Define initial condition
    x = SpatialCoordinate(mesh)
    u_init = Expression('exp(-2*pow(x[0] - 1, 2))', degree=2)
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

    def update(frame, u_n, u, u_, t, dt):
        u_.assign(u_n)  # Update the non-linear term with the previous solution
        solve(a == L, u_)
        u_n.assign(u_)
        t += dt
        line.set_data(x_vals, u_n.compute_vertex_values())
        return line

    ani = animation.FuncAnimation(fig, update, frames=num_steps, fargs=(u_n, u, u_, t, dt),
                                interval=100, blit=True)
    ani.save('vanishing_viscosity.gif', writer='pillow', fps=15)


if __name__ == '__main__':
    solve_1d_burger(epsilon, length, n_x, dt, num_steps)