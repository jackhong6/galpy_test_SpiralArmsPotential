from galpy.potential import SpiralArmsPotential as spiral
import numpy as np
from numpy.testing import assert_allclose
from scipy.misc import derivative as deriv
import unittest

pot = spiral(Cs=[1, 2., np.pi/2])


class TestSpiralArmsPotentialHelpers(unittest.TestCase):

    def test_K(self):
        R = 1
        assert_allclose(pot._K(R), pot._ns * pot._N / R / np.sin(pot._alpha))

        R = 1e-6
        assert_allclose(pot._K(R), pot._ns * pot._N / R / np.sin(pot._alpha))

        R = 0.5
        assert_allclose(pot._K(R), pot._ns * pot._N / R / np.sin(pot._alpha))

    def test_B(self):
        R = 1
        assert_allclose(pot._B(R), pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H))

        R = 1e-6
        assert_allclose(pot._B(R), pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H))

        R = 0.3
        assert_allclose(pot._B(R), pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H))

    def test_D(self):
        assert_allclose(pot._D(3), (1. + pot._K(3)*pot._H + 0.3 * pot._K(3)**2 * pot._H**2.) / (1. + 0.3*pot._K(3) * pot._H))
        assert_allclose(pot._D(1e-6), (1. + pot._K(1e-6)*pot._H + 0.3 * pot._K(1e-6)**2 * pot._H**2.) / (1. + 0.3*pot._K(1e-6) * pot._H))
        assert_allclose(pot._D(.5), (1. + pot._K(.5)*pot._H + 0.3 * pot._K(.5)**2 * pot._H**2.) / (1. + 0.3*pot._K(.5) * pot._H))

    def test_dK_dR(self):
        dx = 1e-8
        assert_allclose(pot._dK_dR(3), deriv(pot._K, 3, dx=dx))
        assert_allclose(pot._dK_dR(2.3), deriv(pot._K, 2.3, dx=dx))
        assert_allclose(pot._dK_dR(-2.3), deriv(pot._K, -2.3, dx=dx))

    def test_dB_dR(self):
        dx = 1e-8
        assert_allclose(pot._dB_dR(3.3), deriv(pot._B, 3.3, dx=dx))
        assert_allclose(pot._dB_dR(1e-3), deriv(pot._B, 1e-3, dx=dx))
        assert_allclose(pot._dB_dR(3), deriv(pot._B, 3, dx=dx))

    def test_dD_dR(self):
        dx = 1e-8
        assert_allclose(pot._dD_dR(2.1), deriv(pot._D, 2.1, dx=dx))
        assert_allclose(pot._dD_dR(1e-3), deriv(pot._D, 1e-3, dx=dx))
        assert_allclose(pot._dD_dR(2), deriv(pot._D, 2, dx=dx))

    def test_d2K_dR2(self):
        dx = 1e-8
        assert_allclose(pot._d2K_dR2(3), deriv(pot._dK_dR, 3, dx=dx))
        assert_allclose(pot._d2K_dR2(-3), deriv(pot._dK_dR, -3, dx=dx))
        assert_allclose(pot._d2K_dR2(.3), deriv(pot._dK_dR, .3, dx=dx))

    def test_d2B_dR2(self):
        dx = 1e-8
        assert_allclose(pot._d2B_dR2(3), deriv(pot._dB_dR, 3, dx=dx))
        assert_allclose(pot._d2B_dR2(-3), deriv(pot._dB_dR, -3, dx=dx))
        assert_allclose(pot._d2B_dR2(.3), deriv(pot._dB_dR, .3, dx=dx))

    def test_d2D_dR2(self):
        dx = 1e-8
        assert_allclose(pot._d2D_dR2(3), deriv(pot._dD_dR, 3, dx=dx))
        assert_allclose(pot._d2D_dR2(1e-3), deriv(pot._dD_dR, 1e-3, dx=dx))
        assert_allclose(pot._d2D_dR2(.3), deriv(pot._dD_dR, .3, dx=dx))

    def test_dK2_dR(self):
        dx = 1e-8
        assert_allclose(pot._dK2_dR(3), deriv(lambda x: pot._K(x) ** 2, 3, dx=dx))
        assert_allclose(pot._dK2_dR(1e-3), deriv(lambda x: pot._K(x) ** 2, 1e-3, dx=dx), rtol=1e-5)
        assert_allclose(pot._dK2_dR(.3), deriv(lambda x: pot._K(x) ** 2, .3, dx=dx))

    def test_dB2_dR(self):
        dx = 1e-8
        assert_allclose(pot._dB2_dR(3), deriv(lambda x: pot._B(x) ** 2, 3, dx=dx))
        assert_allclose(pot._dB2_dR(1e-3), deriv(lambda x: pot._B(x) ** 2, 1e-3, dx=dx), rtol=1e-5)
        assert_allclose(pot._dB2_dR(.3), deriv(lambda x: pot._B(x) ** 2, .3, dx=dx))

    def test_dD2_dR(self):
        dx = 1e-8
        assert_allclose(pot._dD2_dR(3), deriv(lambda x: pot._D(x) ** 2, 3, dx=dx))
        assert_allclose(pot._dD2_dR(1e-3), deriv(lambda x: pot._D(x) ** 2, 1e-3, dx=dx), rtol=1e-4)
        assert_allclose(pot._dD2_dR(.3), deriv(lambda x: pot._D(x) ** 2, .3, dx=dx))

    def test_gamma(self):
        R, phi, t = 1, 2, 3
        assert_allclose(pot._gamma(R, phi, t), pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha) + pot._omega * t))

        R , phi, t = .1, -.2, -.3
        assert_allclose(pot._gamma(R, phi, t), pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha) + pot._omega * t))

        R, phi, t = 0.01, 0, 0
        assert_allclose(pot._gamma(R, phi, t), pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha) + pot._omega * t))

    def test_dgamma_dR(self):
        dx = 1e-8
        assert_allclose(pot._dgamma_dR(3.), deriv(lambda x: pot._gamma(x, 1, 2), 3., dx=dx))
        assert_allclose(pot._dgamma_dR(3), deriv(lambda x: pot._gamma(x, 1, 2), 3, dx=dx))
        assert_allclose(pot._dgamma_dR(0.01), deriv(lambda x: pot._gamma(x, 1, 2), 0.01, dx=dx))

    def test_d2gamma_dR2(self):
        dx = 1e-8
        assert_allclose(pot._d2gamma_dR2(1), deriv(pot._dgamma_dR, 1, dx=dx))
        assert_allclose(pot._d2gamma_dR2(.1), deriv(pot._dgamma_dR, .1, dx=dx))
        assert_allclose(pot._d2gamma_dR2(5), deriv(pot._dgamma_dR, 5, dx=dx))
        assert_allclose(pot._d2gamma_dR2(1e-3), deriv(pot._dgamma_dR, 1e-3, dx=dx))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpiralArmsPotentialHelpers)
    unittest.TextTestRunner(verbosity=2).run(suite)
