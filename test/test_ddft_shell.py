import numpy as np

from analysis import Analysis
from ddft_shell import DDFTShell
from diffusion import Diffusion
from fexc.fexc import Fexc


def test_constant_input():
    dr = 1
    n = 8
    rho_bulk = 1.
    dt = .1
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) * rho_bulk

    ana = Analysis(dr, n)
    f_exc = Fexc(ana)
    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0), rho_bulk)

    rho_s_1, rho_d_1, j_s_1, j_d_1 = ddft.step()

    j_s_expected = np.zeros(n)
    j_d_expected = np.zeros(n)
    np.testing.assert_almost_equal(j_s_expected, j_s_1)
    np.testing.assert_almost_equal(j_d_expected, j_d_1)
    np.testing.assert_almost_equal(rho_s_0, rho_s_1)
    np.testing.assert_almost_equal(rho_d_0, rho_d_1)

def test_delta_input():
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

    rho_s_1, rho_d_1, j_s_1, j_d_1 = ddft.step()

    rho_s_expected = np.asarray([np.exp((4 * np.log(0.4) - np.log(0.025)) / 3), 0.4, 0.025, 0, 0, 0, 0, 0])
    rho_d_expected = np.asarray([np.exp((4 * np.log(0.6) - np.log(0.975)) / 3), 0.6, 0.975, 1, 1, 1, 1, 1])
    j_s_expected = np.asarray([0, 0.25, 0.25, 0, 0, 0, 0, 0])
    j_d_expected = np.asarray([-0.0515749, -0.3015749, -0.25 , 0, 0, 0, 0, 0])
    np.testing.assert_almost_equal(rho_s_expected, rho_s_1)
    np.testing.assert_almost_equal(rho_d_expected, rho_d_1)
    np.testing.assert_almost_equal(j_s_expected, j_s_1)
    np.testing.assert_almost_equal(j_d_expected, j_d_1)

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
