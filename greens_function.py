from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Define the mesh (domain D) and function space
mesh = UnitSquareMesh(50, 50)  # Example: a unit square domain discretized into a 50x50 mesh
V = FunctionSpace(mesh, "P", 1)  # Use linear Lagrange elements

# Define boundary condition: u = 0 on boundary
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), boundary)

# Define the point of singularity (the location of the Dirac delta)
xi = Point(0.3, 0.7)  # Example location of the singularity (can be any point within the domain)

# Define the source term as a Dirac delta function with singularity at the point xi
delta = PointSource(V, xi, 1.0)  # 1.0 is the strength of the Dirac delta

# Define the test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

# Define the weak form of the Poisson equation: a(u, v) = L(v)
a = dot(grad(u), grad(v)) * dx
L = Constant(0) * v * dx  # Initial right-hand side is 0 everywhere except at the source

# Assemble system
A, b = assemble_system(a, L, bc)

# Apply the Dirac delta function to the right-hand side
delta.apply(b)

# Solve the system
u_sol = Function(V)
solve(A, u_sol.vector(), b)

# Plot the solution (the Green's function)
p = plot(u_sol, title="Green's function for Poisson equation", mode='color')
plt.colorbar(p)
plt.show()

# Save the solution to a file
vtkfile = File("greens_function.pvd")
vtkfile << u_sol