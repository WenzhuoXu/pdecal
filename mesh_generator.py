import os
import random
import gmsh
import pyvista as pv


def create_variant_shaped_mesh():
    gmsh.initialize()
    gmsh.clear()

    # Define the different mesh sizes to generate
    mesh_sizes = [0.1, 0.05, 0.025, 0.0125]

    # Perform 800 different geometry samples
    for i in range(800):
        # Choose a random geometry
        geometry_type = random.choice(["circle", "ellipse", "triangle"])

        # Choose a random number of points for the geometry
        num_points = random.randint(3, 6)

        # Choose random coordinates for the points
        points = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(num_points)]

        # Choose a random mesh size
        mesh_size = random.choice(mesh_sizes)

        # Create the geometry
        model = gmsh.model
        model.add("geometry")
        factory = model.occ

        # Define the geometry
        point_ids = []
        for point in points:
            point_ids.append(factory.addPoint(*point, 0, mesh_size))
        if geometry_type == "circle":
            factory.addCircleArc(point_ids[0], point_ids[1], point_ids[0])
        elif geometry_type == "ellipse":
            factory.addEllipseArc(point_ids[0], point_ids[1], point_ids[0])
        elif geometry_type == "triangle":
            factory.addLine(point_ids[0], point_ids[1])
            factory.addLine(point_ids[1], point_ids[2])
            factory.addLine(point_ids[2], point_ids[0])

        # Synchronize the geometry
        factory.synchronize()

        # Define the physical groups
        factory.addPhysicalGroup(1, [1], 1)

        # Generate the mesh
        model.mesh.generate(2)

        # Save the mesh
        filename = f"{geometry_type}_{i}.msh"
        model.mesh.write(os.path.join("results", filename))

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


def test_create_variant_shaped_mesh():
    # Generate the meshes
    create_variant_shaped_mesh()

    # Check that the "results" directory exists
    assert os.path.isdir("results")

    # Check that 800 mesh files were generated
    mesh_files = [filename for filename in os.listdir("results") if filename.endswith(".msh")]
    assert len(mesh_files) == 800

    # Check that each mesh file has a unique name
    assert len(set(mesh_files)) == 800

    # Check that each mesh file can be loaded without errors
    for mesh_file in mesh_files:
        mesh = pv.read(os.path.join("results", mesh_file))
        assert isinstance(mesh, pv.UnstructuredGrid)


if __name__ == "__main__":
    create_variant_shaped_mesh()
    visualize_meshes()

    # Uncomment the following line to run the test
    test_create_variant_shaped_mesh()