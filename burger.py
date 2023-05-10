import numpy as np
import matplotlib.pyplot as plt
from fenics import *

def inviscid_burgers(mesh_resolution):
    # Define domain and mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), mesh_resolution, mesh_resolution)

    # Define function space
    V = VectorFunctionSpace(mesh, "CG", 1)

    # Define initial condition
    u_0 = Expression(("sin(2*pi*x[0]) * sin(2*pi*x[1])", "sin(2*pi*x[0]) * sin(2*pi*x[1])"), degree=2)
    u_n = interpolate(u_0, V)

    # Time step size
    dt = 0.01

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    F = (inner(u - u_n, v) * dx + dt * inner(dot(u_n, grad(u)), v) * dx)

    a, L = lhs(F), rhs(F)
    u = Function(V)

    # Time-stepping
    T = 1.0
    t = 0.0
    while t < T:
        # Update time
        t += dt

        # Compute the solution at the new time level
        solve(a == L, u)

        # Update the previous solution
        u_n.assign(u)

    return u

def L2_error(u_exact, u_approx):
    return errornorm(u_exact, u_approx, 'L2')

def main():
    mesh_resolutions = [10, 20, 40, 80]
    errors = []

    for res in mesh_resolutions:
        print(f"Solving for mesh resolution: {res}")
        u_approx = inviscid_burgers(res)
        plt.figure()
        plot(u_approx, title=f"Approximate solution for resolution {res}")
        plt.show()
        plt.close()

        # Analytical solution for comparison
        u_exact = Expression(("sin(2*pi*x[0]) * sin(2*pi*x[1]) * exp(-2*pi*pi*t)", "sin(2*pi*x[0]) * sin(2*pi*x[1]) * exp(-2*pi*pi*t)"), t=1, degree=2)

        error = L2_error(u_exact, u_approx)
        errors.append(error)
        print(f"Error (L2 norm) for resolution {res}: {error}")

    plt.plot(mesh_resolutions, errors, label="Error")
    plt.xlabel("Mesh resolution")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Inviscid Burger's Equation: Error vs Mesh Resolution")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
