import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from fenicstools.Interpolation import interpolate_nonmatching_mesh
from dolfin import *
try:
    from dolfin import XDMFFile, Mesh, MeshValueCollection
    from dolfin.cpp.mesh import MeshFunctionSizet
except ImportError:
    print("Could not import dolfin. Continuing without Dolfin support.")
import h5py
import meshio
from configparser import ConfigParser



def msh2xdmf(mesh_name, dim=2, directory="."):
    """
    Function converting a MSH mesh into XDMF files.
    The XDMF files are:
        - "domain.xdmf": the domain;
        - "boundaries.xdmf": the boundaries physical groups from GMSH;
    """

    # Get the mesh name has prefix
    prefix = mesh_name.split('.')[0]
    # Read the input mesh
    msh = meshio.read("{}/{}".format(directory, mesh_name))
    # Generate the domain XDMF file
    export_domain(msh, dim, directory, prefix)
    # Generate the boundaries XDMF file
    export_boundaries(msh, dim, directory, prefix)
    # Export association table
    export_association_table(msh, prefix, directory)


def export_domain(msh, dim, directory, prefix):
    """
    Export the domain XDMF file as well as the subdomains values.
    """
    # Set cell type
    if dim == 2:
        cell_type = "triangle"
    elif dim == 3:
        cell_type = "tetra"
    # Generate the cell block for the domain cells
    for i in msh.cells:
        if i.type == cell_type:
            data_array = i.data 
    # data_array = [arr for (t, arr) in msh.cells if t == cell_type]
    
    if len(data_array) == 0:
        print("WARNING: No domain physical group found.")
        return
    else:
        # data = np.concatenate(data_array) # Use this expression if more than 1 domain
        data = data_array
    cells = [
        meshio.CellBlock(
            cell_type=cell_type,
            data=data,
        )
    ]
    # Generate the domain cells data (for the subdomains)
    try:
        cell_data = {
            "subdomains": [
                np.concatenate(
                    [
                        msh.cell_data["gmsh:physical"][i]
                        for i, cellBlock in enumerate(msh.cells)
                        if cellBlock.type == cell_type
                    ]
                )
            ]
        }
    except KeyError:
        raise ValueError(
            """
            No physical group found for the domain.
            Define the domain physical group.
                - if dim=2, the domain is a surface
                - if dim=3, the domain is a volume
            """
        )

    # Generate a meshio Mesh for the domain
    domain = meshio.Mesh(
        points=msh.points[:, :dim],
        cells=cells,
        cell_data=cell_data,
    )
    # Export the XDMF mesh of the domain
    meshio.write(
        "{}/{}_{}".format(directory, prefix, "domain.xdmf"),
        domain,
        file_format="xdmf"
    )


def export_boundaries(msh, dim, directory, prefix):
    """
    Export the boundaries XDMF file.
    """
    # Set the cell type
    if dim == 2:
        cell_type = "line"
    elif dim == 3:
        cell_type = "triangle"
    # Generate the cell block for the boundaries cells
    # data_array = [arr for (t, arr) in msh.cells if t == cell_type]
    data_array = []
    for i in msh.cells:
        if i.type == cell_type:
            data_array.append(i.data) 
    if len(data_array) == 0:
        print("WARNING: No boundary physical group found.")
        return
    else:
        data = np.concatenate(data_array)
        # data = data_array
    boundaries_cells = [
        meshio.CellBlock(
            cell_type=cell_type,
            data=data,
        )
    ]
    # Generate the boundaries cells data
    cell_data = {
        "boundaries": [
            np.concatenate(
                [
                    msh.cell_data["gmsh:physical"][i]
                    for i, cellBlock in enumerate(msh.cells)
                    if cellBlock.type == cell_type
                ]
            )
        ]
    }
    # Generate the meshio Mesh for the boundaries physical groups
    boundaries = meshio.Mesh(
        points=msh.points[:, :dim],
        cells=boundaries_cells,
        cell_data=cell_data,
    )
    # Export the XDMF mesh of the lines boundaries
    meshio.write(
        "{}/{}_{}".format(directory, prefix, "boundaries.xdmf"),
        boundaries,
        file_format="xdmf"
    )


def export_association_table(msh, prefix='mesh', directory='.', verbose=True):
    """
    Display the association between the physical group label and the mesh
    value.
    """
    # Create association table
    association_table = {}

    # Display the correspondance
    formatter = "|{:^20}|{:^20}|"
    topbot = "+{:-^41}+".format("")
    separator = "+{:-^20}+{:-^20}+".format("", "")

    # Display
    if verbose:
        print('\n' + topbot)
        print(formatter.format("GMSH label", "MeshFunction value"))
        print(separator)

    for label, arrays in msh.cell_sets.items():
        # Get the index of the array in arrays
        for i, array in enumerate(arrays):
            if array.size != 0:
                index = i
        # Added check to make sure that the association table
        # doesn't try to import irrelevant information.
        if label != "gmsh:bounding_entities":
            value = msh.cell_data["gmsh:physical"][index][0]
            # Store the association table in a dictionnary
            association_table[label] = value
            # Display the association
            if verbose:
                print(formatter.format(label, value))
    if verbose:
        print(topbot)
    # Export the association table
    file_content = ConfigParser()
    file_content["ASSOCIATION TABLE"] = association_table
    file_name = "{}/{}_{}".format(directory, prefix, "association_table.ini")
    with open(file_name, 'w') as f:
        file_content.write(f)

def import_mesh(
        prefix="mesh",
        subdomains=False,
        dim=2,
        directory=".",
):
    """Function importing a dolfin mesh.

    Arguments:
        prefix (str, optional): mesh files prefix (eg. my_mesh.msh,
            my_mesh_domain.xdmf, my_mesh_bondaries.xdmf). Defaults to "mesh".
        subdomains (bool, optional): True if there are subdomains. Defaults to
            False.
        dim (int, optional): dimension of the domain. Defaults to 2.
        directory (str, optional): directory of the mesh files. Defaults to ".".

    Output:
        - dolfin Mesh object containing the domain;
        - dolfin MeshFunction object containing the physical lines (dim=2) or
            surfaces (dim=3) defined in the msh file and the sub-domains;
        - association table
    """
    # Set the file name
    domain = "{}_domain.xdmf".format(prefix)
    boundaries = "{}_boundaries.xdmf".format(prefix)

    # create 2 xdmf files if not converted before
    if not os.path.exists("{}/{}".format(directory, domain)) or \
       not os.path.exists("{}/{}".format(directory, boundaries)):
        msh2xdmf("{}.msh".format(prefix), dim=dim, directory=directory)

    # Import the converted domain
    mesh = Mesh()
    with XDMFFile("{}/{}".format(directory, domain)) as infile:
        infile.read(mesh)
    # Import the boundaries
    boundaries_mvc = MeshValueCollection("size_t", mesh, dim=dim)
    with XDMFFile("{}/{}".format(directory, boundaries)) as infile:
        infile.read(boundaries_mvc, 'boundaries')
    boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
    # Import the subdomains
    if subdomains:
        subdomains_mvc = MeshValueCollection("size_t", mesh, dim=dim)
        with XDMFFile("{}/{}".format(directory, domain)) as infile:
            infile.read(subdomains_mvc, 'subdomains')
        subdomains_mf = MeshFunctionSizet(mesh, subdomains_mvc)
    # Import the association table
    association_table_name = "{}/{}_{}".format(
        directory, prefix, "association_table.ini")
    file_content = ConfigParser()
    file_content.read(association_table_name)
    association_table = dict(file_content["ASSOCIATION TABLE"])
    # Convert the value from string to int
    for key, value in association_table.items():
        association_table[key] = int(value)
    # Return the Mesh and the MeshFunction objects
    if not subdomains:
        return mesh, boundaries_mf, association_table
    else:
        return mesh, boundaries_mf, subdomains_mf, association_table

def heat_equation(mesh, mesh_resolution, random_heat_source):
    # Define domain and mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1

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
    for sim in range(num_simulations):
        # load the mesh
        # prefix = "mesh_{}_res_{}".format(sim, max(mesh_resolutions))
        # mesh, boundaries_mf, association_table = import_mesh(prefix=prefix, subdomains=True, directory="meshes")

        print(f"Simulation {sim + 1}/{num_simulations}")
        random_heat_source = generate_random_heat_source(max(mesh_resolutions))

        for res in mesh_resolutions:
            print(f"Solving for mesh resolution: {res}")
            prefix = "mesh_{}_res_{}".format(sim, res)
            mesh, boundaries_mf, association_table = import_mesh(prefix=prefix, subdomains=False, directory="mesh")
            u_approx = steady_state_heat_equation(res, random_heat_source, mesh)

            plt.close()
            plt.figure()
            plot(u_approx)
            plt.savefig(f"u_sim_{sim}_res_{res}.png")

            # Save the solution to an XDMF file
            with XDMFFile(f"u_sim_{sim}_res_{res}.xdmf") as xdmf_file:
                xdmf_file.write_checkpoint(u_approx, "u", 0, append=False)

    return meshes                       

def merge_xdmf_files_to_h5(num_simulations, mesh_resolutions, meshes):
    for res in mesh_resolutions:
        mesh = meshes[mesh_resolutions.index(res)]
        mesh_interpolate = meshes[3]
        
        V = FunctionSpace(mesh, "CG", 1)
        Q = FunctionSpace(mesh_interpolate, "CG", 1) 
        
        with h5py.File(f"heat_solutions_res_{res}.h5", "w") as h5_file:
            for sim in range(num_simulations):
                u = Function(V)
                ul = Function(Q)
                with XDMFFile(f"u_sim_{sim}_res_{res}.xdmf") as xdmf_file:
                    xdmf_file.read_checkpoint(u, "u", 0)
                
                ul = interpolate_nonmatching_mesh(u, Q)

                u_array = ul.compute_vertex_values(mesh_interpolate)
                # u_array = u.compute_vertex_values(mesh)

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
    num_simulations = 800
    mesh_resolutions = [10, 20, 40, 80]
    meshes = main_steady_state(mesh_resolutions, num_simulations)
    
    merge_xdmf_files_to_h5(num_simulations, mesh_resolutions, meshes)
    save_mesh_to_xdmf(mesh_resolutions, meshes)
