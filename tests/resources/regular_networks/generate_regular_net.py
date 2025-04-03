import numpy as np
import pandas as pd
import os

from ltrace.slicer.data_utils import dataFrameToTableNode
from ltrace.pore_networks.functions import geo2pnf


def generate_regular_pore_network(radius=5, threshold=0.25, grid_size=(10, 10, 10)):
    np.random.seed(42)

    nx, ny, nz = grid_size
    num_pores = nx * ny * nz

    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T * 2 * radius

    pores = pd.DataFrame(
        {
            "pore.subresolution_porosity": np.ones(num_pores),
            "pore.region_volume": np.ones(num_pores) * 4 * np.pi * radius**3 / 3,
            "pore.equivalent_diameter": np.ones(num_pores) * radius * 2,
            "pore.inscribed_diameter": np.ones(num_pores) * radius * 2,
            "pore.extended_diameter": np.ones(num_pores) * radius * 2,
            "pore.volume": np.random.rand(num_pores) * 4 * np.pi * radius**3 / 3,
            "pore.surface_area": np.random.rand(num_pores) * 4 * np.pi * radius**2,
            "pore.all": np.ones(num_pores, dtype=int),
            "pore.region_label": np.arange(num_pores, dtype=int),
            "pore.phase": np.ones(num_pores, dtype=int),
            "pore.phase1": np.ones(num_pores, dtype=bool),
            "pore.phase2": np.zeros(num_pores, dtype=bool),
            "pore.boundary": np.random.choice([True, False], size=num_pores),  # aqui
            "pore.coords_0": coords[:, 0],
            "pore.coords_1": coords[:, 1],
            "pore.coords_2": coords[:, 2],
            "pore.radius": np.ones(num_pores) * radius,
            "pore.shape_factor": 0.09 * np.random.rand(num_pores),
            "pore.effective_volume": np.ones(num_pores) * 4 * np.pi * radius**3 / 3,
        }
    )

    # Determine boundary thresholds
    threshold_x = int(threshold * nx)
    threshold_y = int(threshold * ny)
    threshold_z = int(threshold * nz)

    pores["pore.xmin"] = pores["pore.coords_0"] < threshold_x * 2 * radius
    pores["pore.xmax"] = pores["pore.coords_0"] >= (nx - threshold_x) * 2 * radius
    pores["pore.ymin"] = pores["pore.coords_1"] < threshold_y * 2 * radius
    pores["pore.ymax"] = pores["pore.coords_1"] >= (ny - threshold_y) * 2 * radius
    pores["pore.zmin"] = pores["pore.coords_2"] < threshold_z * 2 * radius
    pores["pore.zmax"] = pores["pore.coords_2"] >= (nz - threshold_z) * 2 * radius

    # Generate throats by connecting adjacent pores in the structured grid
    connections = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                index = i * ny * nz + j * nz + k
                if i < nx - 1:
                    connections.append((index, index + ny * nz))
                if j < ny - 1:
                    connections.append((index, index + nz))
                if k < nz - 1:
                    connections.append((index, index + 1))

    num_throats = len(connections)
    throats = pd.DataFrame(
        {
            "throat.inscribed_diameter": np.ones(num_throats) * 2 * radius,
            "throat.total_length": np.ones(num_throats) * 4 * radius,
            "throat.direct_length": np.ones(num_throats) * 4 * radius,
            "throat.perimeter": np.ones(num_throats) * 2 * np.pi * radius,
            "throat.cross_sectional_area": np.ones(num_throats) * np.pi * radius**2,
            "throat.equivalent_diameter": np.ones(num_throats) * 2 * radius,
            "throat.all": np.ones(num_throats, dtype=int),
            "throat.conns_0": [c[0] for c in connections],
            "throat.conns_1": [c[1] for c in connections],
            "throat.conns_0_length": np.ones(num_throats) * 4 * radius,
            "throat.conns_1_length": np.ones(num_throats) * 4 * radius,
            "throat.mid_length": np.ones(num_throats) * 4 * radius,
            "throat.phases_0": np.ones(num_throats, dtype=int),
            "throat.phases_1": np.ones(num_throats, dtype=int),
            "throat.shape_factor": 0.09 * np.random.rand(num_throats),
            "throat.volume": np.random.rand(num_throats) * np.pi * radius**2 * 4 * radius,
            "throat.subresolution_porosity": 1.0,
        }
    )

    return pores, throats


for num_pores in 128, 256, 512, 1024, 2048, 4096, 8192, 16384:
    particles_in_side = int(num_pores ** (1.0 / 3.0))
    pores, throats = generate_regular_pore_network(
        radius=5, threshold=1.0 / particles_in_side, grid_size=(particles_in_side,) * 3
    )

    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    folderItemID = shNode.CreateFolderItem(shNode.GetSceneItemID(), str(num_pores))

    pore_node = dataFrameToTableNode(pores)
    pore_node.SetName("pore_table")
    pore_node.SetAttribute("table_type", "pore_table")
    pore_node.SetAttribute("x_size", "10")
    pore_node.SetAttribute("y_size", "10")
    pore_node.SetAttribute("z_size", "10")
    shNode.CreateItem(folderItemID, pore_node)

    throat_node = dataFrameToTableNode(throats)
    throat_node.SetName("throat_table")
    throat_node.SetAttribute("table_type", "throat_table")
    shNode.CreateItem(folderItemID, throat_node)

    statoil_dict = geo2pnf(
        pore_node,
        lambda x: x,
        subres_shape_factor=0.04,
        subres_porositymodifier=1.0,
    )

    statoil_file_strings = {}
    for name in ("link1", "link2", "link3", "node1", "node2", "node3"):
        statoil_file_strings[name] = "\n".join(statoil_dict[name]) + "\n"

    dir_path = str(num_pores)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

    link1 = statoil_file_strings["link1"]
    link2 = statoil_file_strings["link2"]
    link3 = statoil_file_strings["link3"]
    node1 = statoil_file_strings["node1"]
    node2 = statoil_file_strings["node2"]
    node3 = statoil_file_strings["node3"]
    link1_file = open(f"{str(num_pores)}/Image_link1.dat", "w")
    link2_file = open(f"{str(num_pores)}/Image_link2.dat", "w")
    node1_file = open(f"{str(num_pores)}/Image_node1.dat", "w")
    node2_file = open(f"{str(num_pores)}/Image_node2.dat", "w")
    link1_file.write(link1)
    link2_file.write(link2)
    node1_file.write(node1)
    node2_file.write(node2)
    link1_file.close()
    link2_file.close()
    node1_file.close()
    node2_file.close()
