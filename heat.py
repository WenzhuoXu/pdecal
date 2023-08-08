import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from fenicstools.Interpolation import interpolate_nonmatching_mesh
from dolfin import *
import h5py

def heat_equation(mesh_resolution, random_heat_source):
    # Define domain and mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), mesh_resolution, mesh_resolution)

    # Define function space
    V = FunctionSpace(mesh, "CG", 1)

    # Define initial condition
    x = SpatialCoordinate(mesh)
    u_0 = Expression("exp(-(x[0]*x[0] + x[1]*x[1])/(2*sigma*sigma))", sigma=0.1, degree=2)
    u_n = interpolate(u_0, V)

    # Time step size
    dt = 1.0 / mesh_resolution

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    kappa = Constant(0.1)  # Thermal diffusivity
    F = (u - u_n) * v * dx + kappa * dt * inner(grad(u), grad(v)) * dx - random_heat_source * v * dx

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

def steady_state_heat_equation(mesh_resolution, random_heat_source, mesh):
    # Define domain and mesh
    # xmin, xmax = 0, 1
    # ymin, ymax = 0, 1
    # mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), mesh_resolution, mesh_resolution)

    # Define function space
    V = FunctionSpace(mesh, "CG", 1)

    # Define boundary conditions
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0), boundary)

    x = SpatialCoordinate(mesh)
    heat_source = Expression(random_heat_source, sigma=0.1, degree=2)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    kappa = Constant(0.1)  # Thermal diffusivity
    F = kappa * inner(grad(u), grad(v)) * dx - heat_source * v * dx

    a, L = lhs(F), rhs(F)
    u = Function(V)

    # Solve the steady-state problem
    solve(a == L, u, bc)

    return project(u, V)

def generate_random_heat_source(mesh_resolution):
    source_x = np.random.uniform(0, 1)
    source_y = np.random.uniform(0, 1)

    return f"exp(-(pow((x[0] - {source_x}), 2) + pow((x[1] - {source_y}), 2))/(2*sigma*sigma))"
        
 
def L2_error(u_exact, u_approx):
    return errornorm(u_exact, u_approx, 'L2')

def main():
    mesh_resolutions = [10, 20, 40, 80]
    errors = []

    random_heat_source = generate_random_heat_source(max(mesh_resolutions))

    for res in mesh_resolutions:
        print(f"Solving for mesh resolution: {res}")
        u_approx = heat_equation(res, random_heat_source)
        plt.figure()
        plot(u_approx)
        plt.title(f"Heat Equation: Approximate Solution for Mesh Resolution {res}")
        plt.show()
        plt.close()

        # Analytical solution cannot be determined due to the random heat source
        # Compare with the initial condition as a reference
        u_ref = Expression("exp(-(x[0]*x[0] + x[1]*x[1])/(2*sigma*sigma))", sigma=0.1, degree=2)
        error = L2_error(u_ref, u_approx)
        errors.append(error)
        print(f"Error (L2 norm) for resolution {res}: {error}")

    plt.loglog(mesh_resolutions, errors, "o-", label="Error (L2 norm)")
    plt.xlabel("Mesh resolution")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Heat Equation: Error vs Mesh Resolution")
    plt.grid()
    plt.show()

def main_steady_state(mesh_resolutions, num_simulations):
    # create mesh for all resolutions
    meshes = []
    for res in mesh_resolutions:
        xmin, xmax = 0, 1
        ymin, ymax = 0, 1
        meshes.append(RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), res, res))

    for sim in range(num_simulations):
        print(f"Simulation {sim + 1}/{num_simulations}")
        random_heat_source = generate_random_heat_source(max(mesh_resolutions))

        for res in mesh_resolutions:
            print(f"Solving for mesh resolution: {res}")
            u_approx = steady_state_heat_equation(res, random_heat_source, meshes[mesh_resolutions.index(res)])

            plot(u_approx)

            # Save the solution to an XDMF file
            with XDMFFile(f"u_sim_{sim}_res_{res}.xdmf") as xdmf_file:
                xdmf_file.write_checkpoint(u_approx, "u", 0, append=False)

    return meshes                       

def merge_xdmf_files_to_h5(num_simulations, mesh_resolutions, meshes):
    for res in mesh_resolutions:
        mesh = meshes[mesh_resolutions.index(res)]
        mesh_interpolate = meshes[0]
        
        V = FunctionSpace(mesh, "CG", 1)
        Q = FunctionSpace(mesh_interpolate, "CG", 1) 
        
        with h5py.File(f"heat_solutions_res_{res}.h5", "w") as h5_file:
            for sim in range(num_simulations):
                u = Function(V)
                ul = Function(Q)
                with XDMFFile(f"u_sim_{sim}_res_{res}.xdmf") as xdmf_file:
                    xdmf_file.read_checkpoint(u, "u", 0)
                
                # ul = interpolate_nonmatching_mesh(u, Q)

                # u_array = ul.compute_vertex_values(mesh_interpolate)
                u_array = u.compute_vertex_values(mesh)

                h5_file.create_dataset(f"u_sim_{sim}", data=u_array)
                # os.remove(f"u_sim_{sim}_res_{res}.xdmf")

            # randomly check if the data is stored correctly
            check_num = np.random.randint(0, num_simulations)
            print(f"Checking if data for simulation {check_num} is stored correctly")
            try:
                u_array = h5_file[f"u_sim_{check_num}"][:]         
            except KeyError:
                print(f"Data for simulation {check_num} is not stored correctly")

def save_mesh_to_xdmf(mesh_resolutions, meshes):
    # xmin, xmax = 0, 1
    # ymin, ymax = 0, 1
    # mesh_resolutions = [10, 20, 40, 80]
    for res in mesh_resolutions:
        mesh = meshes[mesh_resolutions.index(res)]
        with h5py.File(f"mesh_res_{res}.h5", "w") as h5_file:
            # reconstruct the mesh
            X = mesh.coordinates()
            # X = [points[:, i] for i in range(2)]
            edges(mesh)
            # define connectivity in COO format
            lines = np.zeros((2 * mesh.num_edges(), 2), dtype=np.int32)
            line_lengths = np.zeros(2 * mesh.num_edges(), dtype=np.float64)
            for i, edge in enumerate(edges(mesh)):
                lines[2*i, :] = edge.entities(0)
                lines[2*i+1, :] = np.flipud(edge.entities(0))
                line_lengths[2*i] = edge.length()
                line_lengths[2*i+1] = edge.length()

            h5_file.create_dataset("X", data=X)
            h5_file.create_dataset("lines", data=lines)
            h5_file.create_dataset("line_lengths", data=line_lengths)

if __name__ == "__main__":
    num_simulations = 1
    mesh_resolutions = [5, 7, 10, 100]
    meshes = main_steady_state(mesh_resolutions, num_simulations)
    
    merge_xdmf_files_to_h5(num_simulations, mesh_resolutions, meshes)
    save_mesh_to_xdmf(mesh_resolutions, meshes)
