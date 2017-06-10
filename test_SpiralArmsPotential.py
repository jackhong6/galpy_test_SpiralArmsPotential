from __future__ import division
from galpy.potential import SpiralArmsPotential as spiral
import numpy as np
from numpy import pi
from numpy.testing import assert_allclose
from scipy.misc import derivative as deriv
from astropy import units as u
import unittest


class TestSpiralArmsPotential(unittest.TestCase):

    def test_constructor(self):
        """Test that constructor initializes and converts units correctly."""
        sp = spiral()  # default values
        assert sp._amp == 1
        assert sp._N == -2  # trick to change to left handed coordinate system
        assert sp._alpha == -0.2
        assert sp._r_ref == 1
        assert sp._phi_ref == 0
        assert sp._Rs == 0.5
        assert sp._H == 0.5
        assert sp._Cs == [1]
        assert sp._omega == 0
        assert sp._rho0 == 1 / (4 * pi)
        assert sp.isNonAxi == True
        assert sp._ro == 8
        assert sp._vo == 220

        sp = spiral(N=3, alpha=10*u.deg, r_ref=1, phi_ref=0, Rs=0.5, H=0.5, Cs=[1], omega=0)
        self.assertEqual(sp._alpha, -10 * pi / 180)

    def test_Rforce(self):
        """Tests Rforce against a numerical derivative -d(Potential) / dR."""
        dx = 1e-8
        rtol = 1e-5  # relative tolerance

        pot = spiral()
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.Rforce(R, z, 0),        -deriv(lambda x: pot(x, z, 0),        R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2.2),   -deriv(lambda x: pot(x, z, pi/2.2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),       -deriv(lambda x: pot(x, z, pi),       R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.7*pi/2), -deriv(lambda x: pot(x, z, 3.7*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.3*pi/2), -deriv(lambda x: pot(x, z, 3.3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
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

        pot = spiral(amp=13, N=1, alpha=0.01, r_ref=1.12, phi_ref=0, Cs=[1, 1.5, 8.], omega=-3)
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

        pot = spiral(N=10, r_ref=15, phi_ref=5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(0.01, 0.), -deriv(lambda x: pot(x, 0.), 0.01, dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2.1),   -deriv(lambda x: pot(x, z, pi/2.1),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 1.3*pi),     -deriv(lambda x: pot(x, z, 1.3*pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, .9*pi),     -deriv(lambda x: pot(x, z, .9*pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.3*pi/2), -deriv(lambda x: pot(x, z, 3.3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2.3), -deriv(lambda x: pot(x, z, pi / 2.3), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 1.1*pi),     -deriv(lambda x: pot(x, z, 1.1*pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.5*pi/2), -deriv(lambda x: pot(x, z, 3.5*pi/2), R, dx=dx), rtol=rtol)

    def test_zforce(self):
        """Test zforce against a numerical derivative -d(Potential) / dz"""
        dx = 1e-8
        rtol = 1e-6  # relative tolerance

        pot = spiral()
        # zforce is zero in the plane of the galaxy
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

        pot = spiral(amp=13, N=3, alpha=-.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2], omega=3)
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 2*pi), rtol=rtol)
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

        pot = spiral(N=1, alpha=-0.2, r_ref=.5, Cs=[1, 1.5], omega=-3)
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(32, 0, pi), rtol=rtol)
        assert_allclose(0, pot.zforce(0.123, 0, 3.33*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -1.5
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2.1), -deriv(lambda x: pot(R, x, 3*pi/2.1), z, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral(N=5, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.4, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.5, 0, pi*1.1), rtol=rtol)
        assert_allclose(0, pot.zforce(0.6, 0, 3*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 37, 1.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

    def test_phiforce(self):
        """Test phiforce against a numerical derivative -d(Potential) / d(phi)."""
        dx = 1e-8
        rtol = 1e-6  # relative tolerance

        pot = spiral()
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = .1, -.3
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3, 7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2.1), -deriv(lambda x: pot(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[5, 9, 13], omega=3)
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
        assert_allclose(pot.phiforce(R, z, 3.2*pi/2), -deriv(lambda x: pot(R, z, x), 3.2*pi/2, dx=dx), rtol=rtol)

        pot = spiral(N=1, alpha=0.01, r_ref=1.12, phi_ref=0, Cs=[1, 1.5, 8.], omega=-.333)
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

        pot = spiral(N=10, r_ref=1.5, phi_ref=5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
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
        R, z = 2.1, .12345
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 2*pi), -deriv(lambda x: pot(R, z, x), 2*pi, dx=dx), rtol=rtol)

    def test_R2deriv(self):
        """Test R2deriv against a numerical derivative -d(Rforce) / dR."""
        dx = 1e-8
        rtol = 1e-6  # relative tolerance

        pot = spiral()
        assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3.1*pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3.1*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 2*pi), -deriv(lambda x: pot.Rforce(x, z, 2*pi), R, dx=dx), rtol=rtol)
        R, z = 5, .9
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral(N=1, alpha=-.3, r_ref=.1, phi_ref=pi, Rs=1, H=1, Cs=[1, 2, 3], omega=3)
        assert_allclose(pot.R2deriv(1e-3, 0.),         -deriv(lambda x: pot.Rforce(x, 0.), 1e-3, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3.1*pi/2), -deriv(lambda x: pot.Rforce(x, z, 3.1*pi/2), R, dx=dx), rtol=rtol)
        R, z = 5, .9
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2.4),     -deriv(lambda x: pot.Rforce(x, z, pi / 2.4), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral(N=7, alpha=.1, r_ref=1, phi_ref=1, Rs=1.1, H=.1, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)], omega=-.3)
        assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi/2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 5, .9
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral(N=4, alpha=pi/2, r_ref=1, phi_ref=1, Rs=.7, H=.77, Cs=[3, 4], omega=-1.3)
        assert_allclose(pot.R2deriv(1e-3, 0.),         -deriv(lambda x: pot.Rforce(x, 0.), 1e-3, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, .33*pi/2), -deriv(lambda x: pot.Rforce(x, z, .33*pi/2), R, dx=dx), rtol=rtol)

    def test_z2deriv(self):
        """Test z2deriv against a numerical derivative -d(zforce) / dz"""
        dx = 1e-8
        rtol = 1e-6  # relative tolerance

        pot = spiral()
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

        pot = spiral(N=3, alpha=-0.3, r_ref=.25, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
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

        pot = spiral(amp=5, N=1, alpha=0.1, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2], omega=3)
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

        pot = spiral(N=1, alpha=1, r_ref=3, phi_ref=pi, Cs=[1, 2], omega=-3)
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
        """Test phi2deriv against a numerical derivative -d(phiforce) / d(phi)."""
        dx = 1e-8
        rtol = 1e-7  # relative tolerance

        pot = spiral()
        R, z = .3, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2.5), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2.5, dx=dx), rtol=rtol)
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

        pot = spiral(amp=13, N=1, alpha=-.3, r_ref=0.5, phi_ref=0.1, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
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

        pot = spiral(amp=13, N=5, alpha=0.1, r_ref=.3, phi_ref=.1, Rs=0.77, H=0.747, Cs=[3, 2], omega=-3)
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

        pot = spiral(amp=11, N=7, alpha=.777, r_ref=7, phi_ref=.7, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
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

    def test_dens(self):
        """Test dens against density obtained using Poisson's equation."""
        rtol = 1e-3  # relative tolerance (this one isn't as precise)

        pot = spiral()
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, 1, .5, forcepoisson=False), pot.dens(1, 1, .5, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(.1, .1, .1, forcepoisson=False), pot.dens(.1, .1, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(33, .777, .747, forcepoisson=False), pot.dens(33, .777, .747, forcepoisson=True), rtol=rtol)

        pot = spiral(N=5, alpha=.3, r_ref=.7, omega=5)
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1.2, 1.2, 1.2, forcepoisson=False), pot.dens(1.2, 1.2, 1.2, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(.1, .1, .1, forcepoisson=False), pot.dens(.1, .1, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(33.3, .007, .747, forcepoisson=False), pot.dens(33.3, .007, .747, forcepoisson=True), rtol=rtol)

        pot = spiral(N=3, alpha=.24, r_ref=1, phi_ref=pi, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)], omega=-3)
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, 1, 1, forcepoisson=False), pot.dens(1, 1, 1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(.1, .1, .1, forcepoisson=False), pot.dens(.1, .1, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(3.33, -7.77, -.747, forcepoisson=False), pot.dens(3.33, -7.77, -.747, forcepoisson=True), rtol=rtol)

        pot = spiral(N=4, alpha=pi/2, r_ref=1, phi_ref=1, Rs=7, H=77, Cs=[3, 1, 1], omega=-1.3)
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(3, 2, pi, forcepoisson=False), pot.dens(3, 2, pi, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(.1, .123, .1, forcepoisson=False), pot.dens(.1, .123, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(333, -.777, .747, forcepoisson=False), pot.dens(333, -.777, .747, forcepoisson=True), rtol=rtol)

    def test_Rzderiv(self):
        """Test Rzderiv against a numerical derivative."""
        dx = 1e-8
        rtol = 1e-6

        pot = spiral()
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi/3, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi/4.2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=7, alpha=.1, r_ref=1.123, phi_ref=.3, Rs=0.777, H=.5, Cs=[4.5], omega=-3.4)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, 0.333, pi/3, 0.
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi/4.2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 2, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

        pot = spiral(amp=11, N=2, alpha=.777, r_ref=7, Cs=[8.], omega=0.1)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi/12, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi/4.2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 2, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

        pot = spiral(amp=2, N=1, alpha=-0.1, r_ref=5, Rs=5, H=.7, Cs=[3.5], omega=3)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 0.77, 0.3, pi/3, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3.1, -0.3, pi/5, 2
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

    def test_Rphideriv(self):
        """Test Rphideriv against a numerical derivative."""
        dx = 1e-8
        rtol = 5e-5

        pot = spiral()
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi / 3, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi / 4.2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3 * pi / 2, 5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)

        pot = spiral(N=3, alpha=.21, r_ref=.5, phi_ref=pi, Cs=[2.], omega=-3)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi / 3, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi / 4.2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3 * pi / 2, 5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 2, 1, 100
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.12, 0, 2, 343
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)

    def test_OmegaP(self):
        sp = spiral()
        assert sp.OmegaP() == 0

        sp = spiral(N=1, alpha=2, r_ref=.1, phi_ref=.5, Rs=0.2, H=0.7, Cs=[1,2], omega=-123)
        assert sp.OmegaP() == -123

        sp = spiral(omega=123.456)
        assert sp.OmegaP() == 123.456

    def test_K(self):
        pot = spiral()
        R = 1
        assert_allclose([pot._K(R)], [pot._ns * pot._N / R / np.sin(pot._alpha)])

        R = 1e-6
        assert_allclose([pot._K(R)], [pot._ns * pot._N / R / np.sin(pot._alpha)])

        R = 0.5
        assert_allclose([pot._K(R)], [pot._ns * pot._N / R / np.sin(pot._alpha)])

    def test_B(self):
        pot = spiral()

        R = 1
        assert_allclose([pot._B(R)], [pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H)])

        R = 1e-6
        assert_allclose([pot._B(R)], [pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H)])

        R = 0.3
        assert_allclose([pot._B(R)], [pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H)])

    def test_D(self):
        pot = spiral()

        assert_allclose([pot._D(3)], [(1. + pot._K(3)*pot._H + 0.3 * pot._K(3)**2 * pot._H**2.) / (1. + 0.3*pot._K(3) * pot._H)])
        assert_allclose([pot._D(1e-6)], [(1. + pot._K(1e-6)*pot._H + 0.3 * pot._K(1e-6)**2 * pot._H**2.) / (1. + 0.3*pot._K(1e-6) * pot._H)])
        assert_allclose([pot._D(.5)], [(1. + pot._K(.5)*pot._H + 0.3 * pot._K(.5)**2 * pot._H**2.) / (1. + 0.3*pot._K(.5) * pot._H)])

    def test_dK_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dK_dR(3), deriv(pot._K, 3, dx=dx))
        assert_allclose(pot._dK_dR(2.3), deriv(pot._K, 2.3, dx=dx))
        assert_allclose(pot._dK_dR(-2.3), deriv(pot._K, -2.3, dx=dx))

    def test_dB_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dB_dR(3.3), deriv(pot._B, 3.3, dx=dx))
        assert_allclose(pot._dB_dR(1e-3), deriv(pot._B, 1e-3, dx=dx))
        assert_allclose(pot._dB_dR(3), deriv(pot._B, 3, dx=dx))

    def test_dD_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dD_dR(2.1), deriv(pot._D, 2.1, dx=dx))
        assert_allclose(pot._dD_dR(1e-3), deriv(pot._D, 1e-3, dx=dx))
        assert_allclose(pot._dD_dR(2), deriv(pot._D, 2, dx=dx))

    def test_d2K_dR2(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._d2K_dR2(3), deriv(pot._dK_dR, 3, dx=dx))
        assert_allclose(pot._d2K_dR2(-3), deriv(pot._dK_dR, -3, dx=dx))
        assert_allclose(pot._d2K_dR2(.3), deriv(pot._dK_dR, .3, dx=dx))

    def test_d2B_dR2(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._d2B_dR2(3), deriv(pot._dB_dR, 3, dx=dx))
        assert_allclose(pot._d2B_dR2(-3), deriv(pot._dB_dR, -3, dx=dx))
        assert_allclose(pot._d2B_dR2(.3), deriv(pot._dB_dR, .3, dx=dx))

    def test_d2D_dR2(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._d2D_dR2(3), deriv(pot._dD_dR, 3, dx=dx))
        assert_allclose(pot._d2D_dR2(1e-3), deriv(pot._dD_dR, 1e-3, dx=dx))
        assert_allclose(pot._d2D_dR2(.3), deriv(pot._dD_dR, .3, dx=dx))

    def test_dK2_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dK2_dR(3), deriv(lambda x: pot._K(x) ** 2, 3, dx=dx))
        assert_allclose(pot._dK2_dR(1e-3), deriv(lambda x: pot._K(x) ** 2, 1e-3, dx=dx), rtol=1e-5)
        assert_allclose(pot._dK2_dR(.3), deriv(lambda x: pot._K(x) ** 2, .3, dx=dx))

    def test_dB2_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dB2_dR(3), deriv(lambda x: pot._B(x) ** 2, 3, dx=dx))
        assert_allclose(pot._dB2_dR(1e-3), deriv(lambda x: pot._B(x) ** 2, 1e-3, dx=dx), rtol=1e-5)
        assert_allclose(pot._dB2_dR(.3), deriv(lambda x: pot._B(x) ** 2, .3, dx=dx))

    def test_dD2_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dD2_dR(3), deriv(lambda x: pot._D(x) ** 2, 3, dx=dx))
        assert_allclose(pot._dD2_dR(1e-3), deriv(lambda x: pot._D(x) ** 2, 1e-3, dx=dx), rtol=1e-4)
        assert_allclose(pot._dD2_dR(.3), deriv(lambda x: pot._D(x) ** 2, .3, dx=dx))

    def test_gamma(self):
        pot = spiral()

        R, phi, t = 1, 2, 3
        assert_allclose(pot._gamma(R, phi, t), [pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha) + pot._omega * t)])

        R , phi, t = .1, -.2, -.3
        assert_allclose(pot._gamma(R, phi, t), [pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha) + pot._omega * t)])

        R, phi, t = 0.01, 0, 0
        assert_allclose(pot._gamma(R, phi, t), [pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha) + pot._omega * t)])

    def test_dgamma_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dgamma_dR(3.), deriv(lambda x: pot._gamma(x, 1, 2), 3., dx=dx))
        assert_allclose(pot._dgamma_dR(3), deriv(lambda x: pot._gamma(x, 1, 2), 3, dx=dx))
        assert_allclose(pot._dgamma_dR(0.01), deriv(lambda x: pot._gamma(x, 1, 2), 0.01, dx=dx))

    def test_d2gamma_dR2(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._d2gamma_dR2(1), deriv(pot._dgamma_dR, 1, dx=dx))
        assert_allclose(pot._d2gamma_dR2(.1), deriv(pot._dgamma_dR, .1, dx=dx))
        assert_allclose(pot._d2gamma_dR2(5), deriv(pot._dgamma_dR, 5, dx=dx))
        assert_allclose(pot._d2gamma_dR2(1e-3), deriv(pot._dgamma_dR, 1e-3, dx=dx))

    def test_array_inputs_in_evaluate_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral()
        array = np.linspace(0, 1, 10)
        self.assertRaises(TypeError, sp, array, array, array, array)
        self.assertRaises(TypeError, sp, array, 1, 2, 3)
        self.assertRaises(TypeError, sp, 1, array, 2, 3)
        self.assertRaises(TypeError, sp, 1, 2, array, 3)
        self.assertRaises(TypeError, sp, 1, 2, 3, array)

    def test_array_inputs_in_Rforce_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral()
        array = np.arange(0, 10, 1)

        self.assertRaises(TypeError, sp.Rforce, array, array, array, array)
        self.assertRaises(TypeError, sp.Rforce, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.Rforce, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.Rforce, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.Rforce, 1, 2, 3, array)

    def test_array_inputs_in_phiforce_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral(Cs=[1, 2, 3])
        array = range(0, 10, 1)

        self.assertRaises(TypeError, sp.phiforce, array, array, array, array)
        self.assertRaises(TypeError, sp.phiforce, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.phiforce, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.phiforce, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.phiforce, 1, 2, 3, array)

    def test_array_inputs_in_zforce_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral(N=3, Cs=[3, 4, 5])
        array = [1, 2, 3]
        self.assertRaises(TypeError, sp.zforce, array, array, array, array)
        self.assertRaises(TypeError, sp.zforce, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.zforce, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.zforce, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.zforce, 1, 2, 3, array)

    def test_array_inputs_in_R2deriv_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral(N=3, Cs=[3, 4, 5])
        array = np.array([1, 2, 3])
        self.assertRaises(TypeError, sp.R2deriv, array, array, array, array)
        self.assertRaises(TypeError, sp.R2deriv, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.R2deriv, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.R2deriv, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.R2deriv, 1, 2, 3, array)

    def test_array_inputs_in_z2deriv_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral(N=3, Cs=[3, 4, 5])
        array = np.zeros(5)
        self.assertRaises(TypeError, sp.z2deriv, array, array, array, array)
        self.assertRaises(TypeError, sp.z2deriv, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.z2deriv, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.z2deriv, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.z2deriv, 1, 2, 3, array)

    def test_array_inputs_in_phi2deriv_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral(N=3, Cs=[3, 4, 5])
        array = np.ones(10)
        self.assertRaises(TypeError, sp.phi2deriv, array, array, array, array)
        self.assertRaises(TypeError, sp.phi2deriv, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.phi2deriv, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.phi2deriv, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.phi2deriv, 1, 2, 3, array)

    def test_array_inputs_in_Rzderiv_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral(N=3, amp=13, phi_ref=0.3)
        array = np.ones(10)
        self.assertRaises(TypeError, sp.Rzderiv, array, array, array, array)
        self.assertRaises(TypeError, sp.Rzderiv, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.Rzderiv, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.Rzderiv, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.Rzderiv, 1, 2, 3, array)

    def test_array_inputs_in_Rphideriv_raises_TypeError(self):
        """Test that TypeError is raised if an array is inputted for R, phi, z, or t"""
        sp = spiral(N=7, amp=7, phi_ref=0.7, alpha=-.7)
        array = [7, 7, 7]
        self.assertRaises(TypeError, sp.Rphideriv, array, array, array, array)
        self.assertRaises(TypeError, sp.Rphideriv, array, 1, 2, 3)
        self.assertRaises(TypeError, sp.Rphideriv, 1, array, 2, 3)
        self.assertRaises(TypeError, sp.Rphideriv, 1, 2, array, 3)
        self.assertRaises(TypeError, sp.Rphideriv, 1, 2, 3, array)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpiralArmsPotential)
    unittest.TextTestRunner(verbosity=2).run(suite)
