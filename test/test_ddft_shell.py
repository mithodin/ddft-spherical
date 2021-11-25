import numpy as np
import pytest

from analysis import Analysis
from ddft_shell import DDFTShell
from fexc.calculate_weights import WeightCalculator
from fexc.fexc import Fexc
from fexc.rosenfeld_q3 import RosenfeldQ3
from fexc.weighted_density import WeightedDensity

precision_ideal = 15
# TODO change this to higher number. atm f_exc does not support this
precision_hard_spheres = 4


def input_data_ideal():
    n = 8
    rho_bulk = 1.

    # case: constant densities
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    # 0th bin is not taken into account
    rho_s_0[0] = np.random.rand(1)
    rho_d_0[0] = np.random.rand(1)

    # expectation: constant densities are unchanged
    yield [
        n,
        rho_bulk,
        rho_s_0,
        rho_d_0,
        None,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n) * rho_bulk
    ]

    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    # case: delta density at r=delta_r
    rho_s_0[1] += 0.5 * rho_bulk
    rho_d_0[1] -= 0.5 * rho_bulk

    j1_d = -0.5 * (1 - np.exp((4 * np.log(0.5) - np.log(1)) / 3))

    # expectation: delta peak spreads, delta hole fills
    yield [
        n,
        rho_bulk,
        rho_s_0,
        rho_d_0,
        None,
        np.asarray([0, 0.25, 0.25, 0, 0, 0, 0, 0]),
        np.asarray([0.25 + j1_d, j1_d, -0.25, 0, 0, 0, 0, 0]),
        np.asarray([np.exp((4 * np.log(0.4) - np.log(0.025)) / 3), 0.4, 0.025, 0, 0, 0, 0, 0]),
        np.asarray([np.exp((4 * np.log(0.6) - np.log(0.975)) / 3), 0.6, 0.975, 1, 1, 1, 1, 1]),
    ]

    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    # case: force to create constant flux through shells
    f_ext = np.arange(n) * np.arange(n)
    f_ext[0] = 1
    f_ext = 1 / f_ext
    f_ext = np.stack((f_ext, f_ext))

    # expectation: density is stationary (except innermost bin)
    yield [
        n,
        rho_bulk,
        rho_s_0,
        rho_d_0,
        f_ext,
        np.zeros(n),
        np.asarray([1, 1, 1/4, 1/9, 1/16, 1/25, 1/36, 1/49]),
        np.asarray([0, 0, 0, 0, 0, 0, 0, 0]),
        np.asarray([np.exp((4 * np.log(0.9) - np.log(1)) / 3), 0.9, 1, 1, 1, 1, 1, 1])
    ]

    # case: alternating density
    rho_s_0 = np.asarray([0, 1, 0, 1, 0, 1, 0, 1]) * rho_bulk
    rho_d_0 = rho_bulk - rho_s_0

    # expectation: sum of densities unchanged
    yield [
        n,
        rho_bulk,
        rho_s_0,
        rho_d_0,
        None,
        np.asarray([0,  0.5, 0, 0, 0, 0, 0, 0]),
        np.asarray([0, -0.5, 0, 0, 0, 0, 0, 0]),
        np.asarray([np.exp((4 * np.log(0.8) - np.log(0.2)) / 3), 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]),
        np.asarray([np.exp((4 * np.log(0.2) - np.log(0.8)) / 3), 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2]),
    ]

    # case: density alternating every second bin
    rho_s_0 = np.asarray([0, 0, 1, 1, 0, 0, 1, 1]) * rho_bulk
    rho_d_0 = rho_bulk - rho_s_0

    j_s_expected = np.asarray([0, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5])
    rho_s_expected = np.asarray([
        np.exp((4 * np.log(0.2) - np.log(0.95)) / 3), 2/10, 1-1/20, 1-4/30, 3/40, 6/50, 1-5/60, 1-8/70
    ])

    # expectation: sum of densities unchanged
    j_d_expected = -j_s_expected
    rho_d_expected = rho_bulk - rho_s_expected
    # 0th bin does not obey this, but is not relevant anyway
    rho_d_expected[0] = np.exp((4 * np.log(0.8) - np.log(0.05)) / 3)

    yield [
        n,
        rho_bulk,
        rho_s_0,
        rho_d_0,
        None,
        j_s_expected,
        j_d_expected,
        rho_s_expected,
        rho_d_expected,
    ]

    # different bulk density
    rho_bulk = 0.5
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    # constant densities are still unchanged
    yield [
        n,
        rho_bulk,
        rho_s_0,
        rho_d_0,
        None,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n) * rho_bulk
    ]


@pytest.mark.parametrize("input_data", input_data_ideal())
def test_ideal_gas(input_data):
    n, rho_bulk, rho_s_0, rho_d_0, f_ext, j_s_expected, j_d_expected, rho_s_expected, rho_d_expected = input_data

    dr = 1
    dt = .1

    ana = Analysis(dr, n)
    f_exc = Fexc(ana)
    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0), rho_bulk)

    rho_s_1, rho_d_1, j_s_1, j_d_1 = ddft.step(f_ext=f_ext)

    np.testing.assert_almost_equal(j_s_expected, j_s_1, decimal=precision_ideal)
    np.testing.assert_almost_equal(j_d_expected, j_d_1, decimal=precision_ideal)
    np.testing.assert_almost_equal(rho_s_expected, rho_s_1, decimal=precision_ideal)
    np.testing.assert_almost_equal(rho_d_expected, rho_d_1, decimal=precision_ideal)


def input_data_hard_spheres():
    n = 4096
    rho_bulk = 1.

    # constant densities
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    # constant densities are unchanged
    yield [
        n,
        rho_bulk,
        rho_s_0,
        rho_d_0,
        None,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n) * rho_bulk
    ]


@pytest.mark.parametrize("input_data", input_data_hard_spheres())
def test_hard_spheres(input_data):
    n, rho_bulk, rho_s_0, rho_d_0, f_ext, j_s_expected, j_d_expected, rho_s_expected, rho_d_expected = input_data

    dr = 1/128
    dt = .1

    ana = Analysis(dr, n)
    wc = WeightCalculator()
    wd = WeightedDensity(ana, wc)
    f_exc = RosenfeldQ3(ana, wd)
    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0), rho_bulk)

    rho_s_1, rho_d_1, j_s_1, j_d_1 = ddft.step(f_ext=f_ext)

    np.testing.assert_almost_equal(j_s_expected[1:], j_s_1[1:], decimal=precision_hard_spheres)
    np.testing.assert_almost_equal(j_d_expected[1:], j_d_1[1:], decimal=precision_hard_spheres)
    np.testing.assert_almost_equal(rho_s_expected, rho_s_1, decimal=precision_hard_spheres)
    np.testing.assert_almost_equal(rho_d_expected, rho_d_1, decimal=precision_hard_spheres)


def test_to_shell_and_back():
    dr = 1
    n = 8
    rho_bulk = 1.
    dt = .1
    rho_s_0 = np.arange(n)
    rho_d_0 = n - np.arange(n)

    ana = Analysis(dr, n)
    f_exc = Fexc(ana)
    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0), rho_bulk)

    to_shell_and_back_s = ddft._to_volume_density(ddft._to_shell_density(rho_s_0))
    to_shell_and_back_d = ddft._to_volume_density(ddft._to_shell_density(rho_d_0))

    np.testing.assert_almost_equal(rho_s_0[1:], to_shell_and_back_s[1:])
    np.testing.assert_almost_equal(rho_d_0[1:], to_shell_and_back_d[1:])
