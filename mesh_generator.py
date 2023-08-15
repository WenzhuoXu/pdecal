import os
import meshio
import numpy as np
import random
from configparser import ConfigParser
import gmsh
import pyvista as pv


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


def create_variant_shaped_mesh():
    # Define the different mesh sizes to generate
    mesh_sizes = [0.1, 0.05, 0.025, 0.0125]

    # Perform 800 different geometry samples
    for i in range(800):
        # Choose a random geometry
        geometry_type = random.choice(["circle", "ellipse", "triangle"])

        # Choose a random number of points for the geometry
        num_points = random.randint(3, 6)

        # Choose random coordinates for the points
        points = [(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(num_points)]
        # print(points[0])

        # Choose random radii for the shape
        if geometry_type == "circle":
            radius = random.uniform(1, 2)
        elif geometry_type == "ellipse":
            radius = [random.uniform(1, 2), random.uniform(1, 2)]
            radius_x = np.max(radius)
            radius_y = np.min(radius)

        for res in mesh_sizes:
            gmsh.initialize()
            gmsh.clear()
            # Create the geometry
            model = gmsh.model
            model.add("geometry")
            factory = model.occ

            # Define the geometry
            point_ids = []
            for num in range(num_points):
                point = factory.addPoint(points[num][0], points[num][1], 0, res)
                point_ids.append(point)
            if geometry_type == "circle":
                # add a control point on circle boundary to control the mesh size
                control_point = factory.addPoint(points[0][0] + radius, points[0][1], 0, res)
                circle = factory.addCircle(points[0][0], points[0][1], 0, radius)
                curve_loop = factory.addCurveLoop([circle])
                plane_surface = factory.addPlaneSurface([curve_loop])
                factory.mesh.setSize(factory.getEntities(0), res)
                factory.synchronize()
                model.addPhysicalGroup(2, [plane_surface], 1, 'domain')
                model.addPhysicalGroup(1, [circle], 1, 'boundary')

            elif geometry_type == "ellipse":
                # add a control point on ellipse boundary to control the mesh size
                control_point = factory.addPoint(points[0][0] + radius_x, points[0][1], 0, res)
                ellipse = factory.addEllipse(points[num][0], points[num][1], 0, radius_x, radius_y)
                curve_loop = factory.addCurveLoop([ellipse])
                plane_surface = factory.addPlaneSurface([curve_loop])
                factory.mesh.setSize(factory.getEntities(0), res)
                factory.synchronize()

                model.addPhysicalGroup(2, [plane_surface], 1, 'domain')
                model.addPhysicalGroup(1, [ellipse], 1, 'boundary')

            elif geometry_type == "triangle":
                factory.addLine(point_ids[0], point_ids[1], 1)
                factory.addLine(point_ids[1], point_ids[2], 2)
                factory.addLine(point_ids[2], point_ids[0], 3)
                factory.addCurveLoop([1, 2, 3], 1)
                plane_surface = factory.addPlaneSurface([1])      
                factory.mesh.setSize(factory.getEntities(0), res)
                factory.synchronize()

                model.addPhysicalGroup(2, [plane_surface], 1, 'domain')
                model.addPhysicalGroup(1, [1, 2, 3], 1, 'boundary')

            # Generate the mesh
            model.mesh.generate(2)
            model.mesh.optimize("Netgen")

            # Save the mesh
            save_path = os.path.join("mesh", "mesh_{}_res_{}.msh".format(i, int(1/res)))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gmsh.write(save_path)

            gmsh.finalize()


def visualize_meshes():
    # Load all the mesh files in the "results" directory
    mesh_files = [filename for filename in os.listdir("results") if filename.endswith(".msh")]
    meshes = [pv.read(os.path.join("results", filename)) for filename in mesh_files]

    # Create a multi-block dataset from the meshes
    dataset = pv.MultiBlock(meshes)

    # Create a plotter and add the dataset
    plotter = pv.Plotter()
    plotter.add_mesh(dataset, opacity=0.5)

    # Set the camera position and show the plot
    plotter.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]
    plotter.show()


def convert_to_xdmf():
    # Load all the mesh files in the "results" directory
    mesh_files = [filename for filename in os.listdir("mesh") if filename.endswith(".msh")]
    for mesh_file in mesh_files:
        msh2xdmf(mesh_file, directory="mesh")


if __name__ == "__main__":
    create_variant_shaped_mesh()
    # visualize_meshes()

    # convert_to_xdmf()
    # Uncomment the following line to run the test
    # test_create_variant_shaped_mesh()