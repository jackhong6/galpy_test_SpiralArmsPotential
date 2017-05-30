import galpy.potential_src.SpiralArmsPotential as spiral
from numpy import pi
from numpy.testing import assert_allclose
from scipy.misc import derivative as deriv
import unittest


class TestSpiralArmsPotential(unittest.TestCase):

    def test_Rforce(self):
        dx = 1e-8
        rtol = 0.0001  # relative tolerance = 0.01%

        pot = spiral.SpiralArmsPotential()
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        #assert_allclose(pot.Rforce(0., 0.), -deriv(lambda x: pot(x, 0.), 0., dx=dx), rtol=rtol)     # R=0 does not work
        assert_allclose(pot.Rforce(0.01, 0.), -deriv(lambda x: pot(x, 0.), 0.01, dx=dx), rtol=rtol) # but R=0.01 does
        R, z = 0.3, 0
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2),   -deriv(lambda x: pot(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(0.01, 0.), -deriv(lambda x: pot(x, 0.), 0.01, dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2),   -deriv(lambda x: pot(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=1, alpha=0.01, r_ref=1.12, phi_ref=0, Cs=[1, 1.5, 8.], omega=-3)
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(0.1, 0.), -deriv(lambda x: pot(x, 0.), 0.1, dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2),   -deriv(lambda x: pot(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=10, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(0.01, 0.), -deriv(lambda x: pot(x, 0.), 0.01, dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2),   -deriv(lambda x: pot(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

    def test_zforce(self):
        dx = 1e-6
        rtol = 0.0001  # relative tolerance = 0.01%

        pot = spiral.SpiralArmsPotential()
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(  0, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 3*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(  0, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 3*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=1, alpha=0.01, r_ref=1.12, phi_ref=0, Cs=[1, 1.5, 8.], omega=-3)
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(  0, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 3*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=10, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(  0, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 3*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

    def test_phiforce(self):
        dx = 1e-6
        rtol = 0.0001  # relative tolerance = 0.01%

        pot = spiral.SpiralArmsPotential()
        assert_allclose(pot.phiforce(0, 0, 0), -deriv(lambda x: pot(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2.1), -deriv(lambda x: pot(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
        assert_allclose(pot.phiforce(0, 0, 0), -deriv(lambda x: pot(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=1, alpha=0.01, r_ref=1.12, phi_ref=0, Cs=[1, 1.5, 8.], omega=-3)
        assert_allclose(pot.phiforce(0, 0, 0), -deriv(lambda x: pot(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),        -deriv(lambda x: pot(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),     -deriv(lambda x: pot(R, z, x),   pi/2,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),       -deriv(lambda x: pot(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3.2*pi/2), -deriv(lambda x: pot(R, z, x), 3.2*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3, 4
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=10, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        assert_allclose(pot.phiforce(0, 0, 0), -deriv(lambda x: pot(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 2.1, .123
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)

    def test_R2deriv(self):
        dx = 1e-6
        rtol = 0.0001  # relative tolerance = 0.01%

        assert_allclose(0, 0)

    def test_z2deriv(self):
        dx = 1e-6
        rtol = 0.0001  # relative tolerance = 0.01%

        pot = spiral.SpiralArmsPotential()
        assert_allclose(pot.z2deriv(0, 0, 0), -deriv(lambda x: pot.zforce(0, x, 0), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1.2, .1
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=9, alpha=-0.3, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        assert_allclose(pot.z2deriv(0, 0, 0), -deriv(lambda x: pot.zforce(0, x, 0), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=7, alpha=0.01, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
        assert_allclose(pot.z2deriv(0, 0, 0), -deriv(lambda x: pot.zforce(0, x, 0), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=10, alpha=1, r_ref=3, phi_ref=pi, Cs=[1, 2], omega=-3)
        assert_allclose(pot.z2deriv(0, 0, 0), -deriv(lambda x: pot.zforce(0, x, 0), 0, dx=dx), rtol=rtol)
        R, z = .7, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 2.1, .99
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

    def test_phi2deriv(self):
        dx = 1e-6
        rtol = 0.0001  # relative tolerance = 0.01%

        pot = spiral.SpiralArmsPotential()
        assert_allclose(pot.phi2deriv(0, 0, 0), -deriv(lambda x: pot.phiforce(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=1, alpha=-.3, r_ref=0.5, phi_ref=0.1, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
        assert_allclose(pot.phi2deriv(0, 0, 0), -deriv(lambda x: pot.phiforce(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3.3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3.3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=10, alpha=0.01, r_ref=-.3, phi_ref=.1, Rs=0.77, H=0.747, Cs=[3, 2, 1], omega=-3)
        assert_allclose(pot.phi2deriv(0, 0, 0), -deriv(lambda x: pot.phiforce(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .3, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral.SpiralArmsPotential(N=7, alpha=.777, r_ref=7, phi_ref=.7, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        assert_allclose(pot.phi2deriv(0, 0, 0), -deriv(lambda x: pot.phiforce(0, 0, x), 0, dx=dx), rtol=rtol)
        R, z = .7, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.33
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1.123, .123
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpiralArmsPotential)
    unittest.TextTestRunner(verbosity=2).run(suite)
