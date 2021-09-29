import numpy as np

from analysis import Analysis
from ddft_shell import DDFTShell
from diffusion import Diffusion
from fexc.fexc import Fexc


def test_constant_input():
    dr = 1
    n = 8
    dt = 1
    rho_s_0 = np.zeros(n)
    rho_d_0 = np.ones(n) *(0.72 + 2/300)
    j_s_0 = np.zeros(n)
    j_d_0 = np.zeros(n)

    ana = Analysis(dr, n)
    f_exc = Fexc(ana)

    ddft = DDFTShell(ana, dt, f_exc, (rho_s_0, rho_d_0))
    rho_s_1, rho_d_1, j_s_1, j_d_1 = ddft.step()

    assert_equals(j_s_0, j_s_1)
    assert_equals(j_d_0, j_d_1)
    assert_equals(rho_s_0, rho_s_1)
    assert_equals(rho_d_0, rho_d_1)

def test_constant_norm():
    dr = 1
    n = 4
    dt = 0.1
    rho_s_0 = np.ones(n) *(0.72 + 2/300)
    rho_d_0 = np.zeros(n)

def assert_equals(x, y):
    assert np.all(x == y), 'failed asserting, that ' + x + ' equals ' + y

# def test_j_exc():
#     dr = 2**-7
#     n = 4096
#     dt = 10**-7
#     ana = Analysis(dr, n)
#     f_exc = Fexc(ana)

#     sigma0 = 5.0
#     gauss = np.exp(-(np.arange(n)*dr/sigma0)**2/2)/sigma0/np.sqrt(2*np.pi)
#     gauss /= ana.integrate(gauss)

#     ddft = DDFTShell(ana, dt, f_exc, (gauss, np.zeros(n)))
#     j_s, j_d = ddft.j_exc()
#     assert np.all(j_s == np.zeros(n))
#     assert np.all(j_d == np.zeros(n))
