"""
地质统计模块测试套件
"""

import numpy as np
import pytest
from reservoirpy.geostatistics.variogram import VariogramModel, VariogramParams
from reservoirpy.geostatistics.perm_generator import PermeabilityGenerator


class TestVariogramParams:
    def test_default_params(self):
        params = VariogramParams()
        assert params.azimuth == 0.0
        assert params.nugget == 0.0
        assert params.major_range == 100.0
        assert params.minor_range == 100.0
        assert params.sill == 1.0
        assert params.vtype == 'exponential'

    def test_to_list(self):
        params = VariogramParams(
            azimuth=45, nugget=0.1, major_range=200,
            minor_range=100, sill=0.8, vtype='spherical',
        )
        lst = params.to_list()
        assert lst == [45, 0.1, 200, 100, 0.8, 'Spherical']

    def test_from_list(self):
        lst = [30, 0.05, 150, 80, 0.9, 'Exponential']
        params = VariogramParams.from_list(lst)
        assert params.azimuth == 30
        assert params.nugget == 0.05
        assert params.major_range == 150
        assert params.minor_range == 80
        assert params.sill == 0.9
        assert params.vtype == 'exponential'

    def test_roundtrip(self):
        params = VariogramParams(azimuth=60, nugget=0.2, major_range=300,
                                 minor_range=150, sill=1.2, vtype='gaussian')
        lst = params.to_list()
        restored = VariogramParams.from_list(lst)
        assert restored.azimuth == params.azimuth
        assert restored.nugget == params.nugget
        assert restored.major_range == params.major_range
        assert restored.minor_range == params.minor_range
        assert restored.sill == params.sill
        assert restored.vtype == params.vtype


class TestVariogramModel:
    def test_creation(self):
        coords = np.random.rand(30, 2) * 200
        values = np.random.randn(30)
        vm = VariogramModel(coords, values)
        assert vm.coords.shape == (30, 2)
        assert vm.values.shape == (30,)

    def test_invalid_coords(self):
        with pytest.raises(ValueError):
            VariogramModel(np.random.rand(30), np.random.randn(30))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            VariogramModel(np.random.rand(30, 2), np.random.randn(20))

    def test_fit(self):
        np.random.seed(42)
        coords = np.random.rand(50, 2) * 200
        values = np.random.randn(50)
        vm = VariogramModel(coords, values)
        params = vm.fit(model='exponential', n_lags=10)
        assert isinstance(params, VariogramParams)
        assert params.vtype == 'exponential'
        assert params.major_range > 0
        assert params.sill > 0

    def test_set_params_manual(self):
        coords = np.random.rand(10, 2) * 100
        values = np.random.randn(10)
        vm = VariogramModel(coords, values)
        params = vm.set_params_manual(
            azimuth=45, nugget=0.1, major_range=200,
            minor_range=100, sill=0.8, vtype='spherical',
        )
        assert params.azimuth == 45
        assert params.vtype == 'spherical'


class TestPermeabilityGenerator:
    def test_creation(self):
        gen = PermeabilityGenerator(nx=10, ny=10, dx=10, dy=10)
        assert gen.nx == 10
        assert gen.ny == 10
        assert gen._pred_grid.shape == (100, 2)

    def test_unconditional_single(self):
        gen = PermeabilityGenerator(nx=5, ny=5, dx=10, dy=10)
        perm = gen.generate(
            major_range=30, minor_range=30,
            sill=1.0, vtype='exponential',
            n_realizations=1, seed=42,
            mean_log_perm=2.0, std_log_perm=0.5,
        )
        assert perm.shape == (1, 5, 5)
        assert np.all(perm > 0)

    def test_unconditional_multiple(self):
        gen = PermeabilityGenerator(nx=5, ny=5, dx=10, dy=10)
        result = gen.generate(
            major_range=30, minor_range=30,
            sill=1.0, vtype='exponential',
            n_realizations=3, seed=42,
            mean_log_perm=2.0, std_log_perm=0.5,
        )
        assert 'perm_fields' in result
        assert result['perm_fields'].shape == (3, 1, 5, 5)

    def test_conditional(self):
        gen = PermeabilityGenerator(nx=5, ny=5, dx=10, dy=10)
        hard_data = np.array([
            [5.0, 5.0, 100.0],
            [45.0, 45.0, 50.0],
        ])
        perm = gen.generate(
            hard_data=hard_data,
            major_range=30, minor_range=30,
            sill=1.0, vtype='exponential',
            n_realizations=1, seed=42,
        )
        assert perm.shape == (1, 5, 5)
        assert np.all(perm > 0)

    def test_generate_from_config(self):
        gen = PermeabilityGenerator(nx=5, ny=5, dx=10, dy=10)
        config = {
            'variogram': {
                'major_range': 30, 'minor_range': 30,
                'sill': 1.0, 'vtype': 'exponential',
            },
            'simulation': {
                'n_realizations': 1, 'seed': 42,
                'mean_log_perm': 2.0, 'std_log_perm': 0.5,
            },
        }
        perm = gen.generate_from_config(config)
        assert perm.shape == (1, 5, 5)
        assert np.all(perm > 0)

    def test_3d_grid(self):
        gen = PermeabilityGenerator(nx=5, ny=5, nz=3, dx=10, dy=10, dz=10)
        perm = gen.generate(
            major_range=30, minor_range=30,
            sill=1.0, vtype='exponential',
            n_realizations=1, seed=42,
            mean_log_perm=2.0, std_log_perm=0.5,
        )
        assert perm.shape == (3, 5, 5)

    def test_reproducibility(self):
        gen = PermeabilityGenerator(nx=5, ny=5, dx=10, dy=10)
        perm1 = gen.generate(
            major_range=30, minor_range=30,
            n_realizations=1, seed=123,
            mean_log_perm=2.0, std_log_perm=0.5,
        )
        perm2 = gen.generate(
            major_range=30, minor_range=30,
            n_realizations=1, seed=123,
            mean_log_perm=2.0, std_log_perm=0.5,
        )
        np.testing.assert_array_almost_equal(perm1, perm2)
