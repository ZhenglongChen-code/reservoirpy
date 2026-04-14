import pytest
import numpy as np
from reservoirpy.core.linear_solver import LinearSolver, solve_linear_system
from scipy.sparse import diags


class TestLinearSolver:
    @pytest.fixture
    def simple_system(self):
        n = 10
        A = diags([-1, 4, -1], [-1, 0, 1], shape=(n, n)).tocsr()
        b = np.ones(n)
        return A, b

    def test_direct_solve(self, simple_system):
        A, b = simple_system
        solver = LinearSolver({'method': 'direct'})
        x = solver.solve(A, b)
        assert np.allclose(A @ x, b, atol=1e-8)

    def test_bicgstab_solve(self, simple_system):
        A, b = simple_system
        solver = LinearSolver({'method': 'bicgstab', 'tolerance': 1e-10})
        x = solver.solve(A, b)
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_cg_solve(self, simple_system):
        A, b = simple_system
        solver = LinearSolver({'method': 'cg', 'tolerance': 1e-10})
        x = solver.solve(A, b)
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_get_info(self):
        solver = LinearSolver({'method': 'bicgstab', 'tolerance': 1e-8})
        info = solver.get_info()
        assert info['method'] == 'bicgstab'
        assert info['tolerance'] == 1e-8

    def test_update_config(self):
        solver = LinearSolver()
        solver.update_config({'method': 'direct'})
        assert solver.method == 'direct'

    def test_module_level_function(self, simple_system):
        A, b = simple_system
        x = solve_linear_system(A, b, method='direct')
        assert np.allclose(A @ x, b, atol=1e-8)
