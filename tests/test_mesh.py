import pytest
import numpy as np
from reservoirpy.mesh.mesh import StructuredMesh, CubeCell, Node, BaseMesh, BaseCell


class TestNode:
    def test_node_creation(self):
        node = Node(1.0, 2.0, 3.0)
        assert node.x == 1.0
        assert node.y == 2.0
        assert node.z == 3.0
        assert node.coord == [1.0, 2.0, 3.0]


class TestCubeCell:
    def test_cell_creation(self):
        cell = CubeCell(index=0, center=[5.0, 5.0, 5.0], volume=1000.0)
        assert cell.index == 0
        assert cell.center == [5.0, 5.0, 5.0]
        assert cell.volume == 1000.0
        assert cell.get_face_count() == 6
        assert cell.get_vertex_count() == 8
        assert len(cell.neighbors) == 6
        assert len(cell.vertices) == 8


class TestStructuredMesh:
    @pytest.fixture
    def mesh(self):
        return StructuredMesh(nx=5, ny=5, nz=1, dx=10.0, dy=10.0, dz=5.0)

    def test_mesh_creation(self, mesh):
        assert mesh.nx == 5
        assert mesh.ny == 5
        assert mesh.nz == 1
        assert mesh.n_cells == 25
        assert mesh.total_cells == 25
        assert mesh.grid_shape == (5, 5, 1)

    def test_node_generation(self, mesh):
        assert len(mesh.node_list) == (5 + 1) * (5 + 1) * (1 + 1)

    def test_cell_generation(self, mesh):
        assert len(mesh.cell_list) == 25

    def test_cell_volume(self, mesh):
        for cell in mesh.cell_list:
            assert cell.volume == pytest.approx(10.0 * 10.0 * 5.0)

    def test_get_cell_index(self, mesh):
        idx = mesh.get_cell_index(0, 2, 3)
        assert idx == 2 * 5 + 3

    def test_get_cell_coords(self, mesh):
        i, j, k = mesh.get_cell_coords(13)
        assert i == 0
        assert j == 2
        assert k == 3

    def test_get_neighbors_ijk(self, mesh):
        neighbors = mesh.get_neighbors(0, 2, 2)
        assert len(neighbors) == 6
        assert neighbors[0] == 2 * 5 + 1
        assert neighbors[1] == 2 * 5 + 3

    def test_get_neighbors_cell_index(self, mesh):
        cell_index = mesh.get_cell_index(0, 2, 2)
        neighbors_by_index = mesh.get_neighbors(cell_index)
        neighbors_by_ijk = mesh.get_neighbors(0, 2, 2)
        assert neighbors_by_index == neighbors_by_ijk

    def test_is_boundary_cell_ijk(self, mesh):
        assert mesh.is_boundary_cell(0, 0, 0) is True
        assert mesh.is_boundary_cell(0, 2, 2) is False
        assert mesh.is_boundary_cell(0, 4, 4) is True

    def test_is_boundary_cell_index(self, mesh):
        corner = mesh.get_cell_index(0, 0, 0)
        center = mesh.get_cell_index(0, 2, 2)
        assert mesh.is_boundary_cell(corner) is True
        assert mesh.is_boundary_cell(center) is False

    def test_3d_mesh(self):
        mesh3d = StructuredMesh(nx=3, ny=3, nz=3, dx=10, dy=10, dz=10)
        assert mesh3d.n_cells == 27
        assert mesh3d.is_boundary_cell(0, 0, 0) is True
        assert mesh3d.is_boundary_cell(1, 1, 1) is False
