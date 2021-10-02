import numpy as np

from analysis import Analysis
from ddft_shell import DDFTShell
from diffusion import Diffusion
from fexc.fexc import Fexc


def test_constant_density():
    dr = 1
    n = 8
    rho_bulk = 1.
    dt = .1
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    ana = Analysis(dr, n)
    f_exc = Fexc(ana)
    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0), rho_bulk)

    rho_s_1, rho_d_1, j_s_1, j_d_1, _, _ = ddft.step()

    j_s_expected = np.zeros(n)
    j_d_expected = np.zeros(n)
    rho_s_expected = rho_s_0
    rho_d_expected = rho_d_0

    np.testing.assert_almost_equal(j_s_expected, j_s_1)
    np.testing.assert_almost_equal(j_d_expected, j_d_1)
    np.testing.assert_almost_equal(rho_s_expected, rho_s_1)
    np.testing.assert_almost_equal(rho_d_expected, rho_d_1)

def test_delta_density():
    dr = 1
    n = 8
    rho_bulk = 1.
    dt = .1
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    d_rho = 0.5 * rho_bulk

    rho_s_0[1] += d_rho
    rho_d_0[1] -= d_rho

    ana = Analysis(dr, n)
    f_exc = Fexc(ana)
    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0), rho_bulk)

    rho_s_1, rho_d_1, j_s_1, j_d_1, _, _ = ddft.step()

    j_s_expected = np.asarray([0, 0.25, 0.25, 0, 0, 0, 0, 0])
    j_d_expected = np.asarray([-0.0515749, -0.3015749, -0.25 , 0, 0, 0, 0, 0])
    rho_s_expected = np.asarray([np.exp((4 * np.log(0.4) - np.log(0.025)) / 3), 0.4, 0.025, 0, 0, 0, 0, 0])
    rho_d_expected = np.asarray([np.exp((4 * np.log(0.6) - np.log(0.975)) / 3), 0.6, 0.975, 1, 1, 1, 1, 1])
    
    np.testing.assert_almost_equal(j_s_expected, j_s_1)
    np.testing.assert_almost_equal(j_d_expected, j_d_1)
    np.testing.assert_almost_equal(rho_s_expected, rho_s_1)
    np.testing.assert_almost_equal(rho_d_expected, rho_d_1)

def test_constant_current():
    dr = 1
    n = 8
    rho_bulk = 1.
    dt = .1
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    ana = Analysis(dr, n)
    f_exc = Fexc(ana)
    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0), rho_bulk)

    f_ext = np.arange(n) * np.arange(n)
    f_ext[0] = 1
    f_ext = 1 / f_ext
    f_ext = np.stack((f_ext, f_ext))

    rho_s_1, rho_d_1, j_s_1, j_d_1, _, _ = ddft.step(f_ext=f_ext)

    j_s_expected = np.zeros(n)
    j_d_expected = np.asarray([1., 1., 0.25, 0.1111111, 0.0625 , 0.04 , 0.0277778, 0.0204082])
    rho_s_expected = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
    rho_d_expected = np.asarray([0.8689404, 0.9, 1, 1, 1, 1, 1, 1])

    np.testing.assert_almost_equal(j_s_expected[1:], j_s_1[1:])
    np.testing.assert_almost_equal(j_d_expected, j_d_1)
    np.testing.assert_almost_equal(rho_s_expected, rho_s_1)
    np.testing.assert_almost_equal(rho_d_expected, rho_d_1)

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

    to_shell_and_back_s = ddft.to_volume_density(ddft.to_shell_density(rho_s_0))
    to_shell_and_back_d = ddft.to_volume_density(ddft.to_shell_density(rho_d_0))

    np.testing.assert_almost_equal(rho_s_0[1:], to_shell_and_back_s[1:])
    np.testing.assert_almost_equal(rho_d_0[1:], to_shell_and_back_d[1:])
