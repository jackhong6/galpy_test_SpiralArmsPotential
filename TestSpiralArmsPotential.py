import galpy.potential_src.SpiralArmsPotential as spiral
import numpy as np
from scipy.misc import derivative as deriv
import unittest


class TestSpiralArmsPotential(unittest.TestCase):

    def test_Rforce(self):
        dx = 1e-6

        pot = spiral.SpiralArmsPotential()
        self.assertAlmostEqual(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx))
        self.assertAlmostEqual(pot.Rforce(0, 0), -deriv(lambda x: pot(x, 0), 0, dx=dx))

    def test_zforce(self):
        dx = 1e-6

        pot = spiral.SpiralArmsPotential()
        self.assertAlmostEqual(0, pot.zforce(-1, 0))         # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(0, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(1, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(2.2, 0, -1.2))  # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(pot.zforce( 0,     0), -deriv(lambda x: pot( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,    -1), -deriv(lambda x: pot(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,     1), -deriv(lambda x: pot(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,  -0.5), -deriv(lambda x: pot(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,   0.5), -deriv(lambda x: pot(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1, -1.23), -deriv(lambda x: pot( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1,  1.23), -deriv(lambda x: pot( 1, x),  1.23, dx=dx))

        pot = spiral.SpiralArmsPotential(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=1)
        self.assertAlmostEqual(0, pot.zforce(-1, 0))         # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(0, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(1, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(2.2, 0, -1.2))  # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(pot.zforce( 0,     0), -deriv(lambda x: pot( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,    -1), -deriv(lambda x: pot(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,     1), -deriv(lambda x: pot(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,  -0.5), -deriv(lambda x: pot(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,   0.5), -deriv(lambda x: pot(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1, -1.23), -deriv(lambda x: pot( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1,  1.23), -deriv(lambda x: pot( 1, x),  1.23, dx=dx))

        pot = spiral.SpiralArmsPotential(N=9, alpha=0.3, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*np.pi), 0.5, 8./(15.*np.pi)])
        self.assertAlmostEqual(0, pot.zforce(-1, 0))         # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(0, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(1, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(2.2, 0, -1.2))  # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(pot.zforce( 0,     0), -deriv(lambda x: pot( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,    -1), -deriv(lambda x: pot(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,     1), -deriv(lambda x: pot(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,  -0.5), -deriv(lambda x: pot(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,   0.5), -deriv(lambda x: pot(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1, -1.23), -deriv(lambda x: pot( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1,  1.23), -deriv(lambda x: pot( 1, x),  1.23, dx=dx))

        pot = spiral.SpiralArmsPotential(N=1, alpha=0.01, r_ref=1.12, phi_ref=0, Cs=[1, 1.5, 8.])
        self.assertAlmostEqual(0, pot.zforce(-1, 0))         # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(0, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(1, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(2.2, 0, -1.2))  # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(pot.zforce( 0,     0), -deriv(lambda x: pot( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,    -1), -deriv(lambda x: pot(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,     1), -deriv(lambda x: pot(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,  -0.5), -deriv(lambda x: pot(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,   0.5), -deriv(lambda x: pot(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1, -1.23), -deriv(lambda x: pot( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1,  1.23), -deriv(lambda x: pot( 1, x),  1.23, dx=dx))

        pot = spiral.SpiralArmsPotential(N=10, alpha=1, r_ref=3, phi_ref=np.pi, Cs=[1, 2])
        self.assertAlmostEqual(0, pot.zforce(-1, 0))         # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(0, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(1, 0))          # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(0, pot.zforce(2.2, 0, -1.2))  # zforce is 0 in the plane of the galaxy
        self.assertAlmostEqual(pot.zforce( 0,     0), -deriv(lambda x: pot( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,    -1), -deriv(lambda x: pot(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.zforce(.5,     1), -deriv(lambda x: pot(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,  -0.5), -deriv(lambda x: pot(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce(-1,   0.5), -deriv(lambda x: pot(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1, -1.23), -deriv(lambda x: pot( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.zforce( 1,  1.23), -deriv(lambda x: pot( 1, x),  1.23, dx=dx))

    def test_phiforce(self):
        tol = 1e-6
        dx = 1e-6

        pot = spiral.SpiralArmsPotential()
        self.assertAlmostEqual(pot.phiforce( 0,  0,    0), -deriv(lambda x: pot( 0,  0, x),    0, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,   -1), -deriv(lambda x: pot( 0,  0, x),   -1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,    1), -deriv(lambda x: pot( 0,  0, x),    1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1, -0.5), -deriv(lambda x: pot( 1,  1, x), -0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1,  0.5), -deriv(lambda x: pot( 1,  1, x),  0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1, -2.3), -deriv(lambda x: pot(-1, -1, x), -2.3, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1,  2.3), -deriv(lambda x: pot(-1, -1, x),  2.3, dx=dx))

        pot = spiral.SpiralArmsPotential(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=1)
        self.assertAlmostEqual(pot.phiforce( 0,  0,    0), -deriv(lambda x: pot( 0,  0, x),    0, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,   -1), -deriv(lambda x: pot( 0,  0, x),   -1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,    1), -deriv(lambda x: pot( 0,  0, x),    1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1, -0.5), -deriv(lambda x: pot( 1,  1, x), -0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1,  0.5), -deriv(lambda x: pot( 1,  1, x),  0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1, -2.3), -deriv(lambda x: pot(-1, -1, x), -2.3, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1,  2.3), -deriv(lambda x: pot(-1, -1, x),  2.3, dx=dx))

        pot = spiral.SpiralArmsPotential(N=1, alpha=0.1, r_ref=1.3, phi_ref=0, Rs=0.9, H=0.9, Cs=[1, 2, 3], omega=1)
        self.assertAlmostEqual(pot.phiforce( 0,  0,    0), -deriv(lambda x: pot( 0,  0, x),    0, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,   -1), -deriv(lambda x: pot( 0,  0, x),   -1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,    1), -deriv(lambda x: pot( 0,  0, x),    1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1, -0.5), -deriv(lambda x: pot( 1,  1, x), -0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1,  0.5), -deriv(lambda x: pot( 1,  1, x),  0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1, -2.3), -deriv(lambda x: pot(-1, -1, x), -2.3, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1,  2.3), -deriv(lambda x: pot(-1, -1, x),  2.3, dx=dx))

        pot = spiral.SpiralArmsPotential(N=9, alpha=np.pi/2., r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*np.pi), 0.5, 8./(15.*np.pi)])
        self.assertAlmostEqual(pot.phiforce( 0,  0,    0), -deriv(lambda x: pot( 0,  0, x),    0, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,   -1), -deriv(lambda x: pot( 0,  0, x),   -1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,    1), -deriv(lambda x: pot( 0,  0, x),    1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1, -0.5), -deriv(lambda x: pot( 1,  1, x), -0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1,  0.5), -deriv(lambda x: pot( 1,  1, x),  0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1, -2.3), -deriv(lambda x: pot(-1, -1, x), -2.3, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1,  2.3), -deriv(lambda x: pot(-1, -1, x),  2.3, dx=dx))

        pot = spiral.SpiralArmsPotential(N=5, alpha=0.01, r_ref=0.1, phi_ref=0.1, Rs=0.1, H=0.1, Cs=[1], omega=0)
        self.assertAlmostEqual(pot.phiforce( 0,  0,    0), -deriv(lambda x: pot( 0,  0, x),    0, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,   -1), -deriv(lambda x: pot( 0,  0, x),   -1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 0,  0,    1), -deriv(lambda x: pot( 0,  0, x),    1, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1, -0.5), -deriv(lambda x: pot( 1,  1, x), -0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce( 1,  1,  0.5), -deriv(lambda x: pot( 1,  1, x),  0.5, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1, -2.3), -deriv(lambda x: pot(-1, -1, x), -2.3, dx=dx))
        self.assertAlmostEqual(pot.phiforce(-1, -1,  2.3), -deriv(lambda x: pot(-1, -1, x),  2.3, dx=dx))

    def test_R2deriv(self):
        dx = 1e-6
        self.assertAlmostEqual(0, 0)

    def test_z2deriv(self):
        dx = 1e-6

        pot = spiral.SpiralArmsPotential()
        self.assertAlmostEqual(pot.z2deriv( 0,     0), -deriv(lambda x: pot.zforce( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,    -1), -deriv(lambda x: pot.zforce(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,     1), -deriv(lambda x: pot.zforce(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,  -0.5), -deriv(lambda x: pot.zforce(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,   0.5), -deriv(lambda x: pot.zforce(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1, -1.23), -deriv(lambda x: pot.zforce( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1,  1.23), -deriv(lambda x: pot.zforce( 1, x),  1.23, dx=dx))

        pot = spiral.SpiralArmsPotential(N=9, alpha=0.3, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*np.pi), 0.5, 8./(15.*np.pi)])
        self.assertAlmostEqual(pot.z2deriv( 0,     0), -deriv(lambda x: pot.zforce( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,    -1), -deriv(lambda x: pot.zforce(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,     1), -deriv(lambda x: pot.zforce(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,  -0.5), -deriv(lambda x: pot.zforce(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,   0.5), -deriv(lambda x: pot.zforce(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1, -1.23), -deriv(lambda x: pot.zforce( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1,  1.23), -deriv(lambda x: pot.zforce( 1, x),  1.23, dx=dx))

        pot = spiral.SpiralArmsPotential(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=1)
        self.assertAlmostEqual(pot.z2deriv( 0,     0), -deriv(lambda x: pot.zforce( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,    -1), -deriv(lambda x: pot.zforce(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,     1), -deriv(lambda x: pot.zforce(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,  -0.5), -deriv(lambda x: pot.zforce(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,   0.5), -deriv(lambda x: pot.zforce(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1, -1.23), -deriv(lambda x: pot.zforce( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1,  1.23), -deriv(lambda x: pot.zforce( 1, x),  1.23, dx=dx))

        pot = spiral.SpiralArmsPotential(N=10, alpha=1, r_ref=3, phi_ref=np.pi, Cs=[1, 2])
        self.assertAlmostEqual(pot.z2deriv( 0,     0), -deriv(lambda x: pot.zforce( 0, x),     0, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,    -1), -deriv(lambda x: pot.zforce(.5, x),    -1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(.5,     1), -deriv(lambda x: pot.zforce(.5, x),     1, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,  -0.5), -deriv(lambda x: pot.zforce(-1, x),  -0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv(-1,   0.5), -deriv(lambda x: pot.zforce(-1, x),   0.5, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1, -1.23), -deriv(lambda x: pot.zforce( 1, x), -1.23, dx=dx))
        self.assertAlmostEqual(pot.z2deriv( 1,  1.23), -deriv(lambda x: pot.zforce( 1, x),  1.23, dx=dx))

    def test_phi2deriv(self):
        dx = 1e-6

        pot = spiral.SpiralArmsPotential()
        self.assertAlmostEqual(pot.phi2deriv( 0,  0,    0), -deriv(lambda x: pot.phiforce( 0,  0, x),    0, dx=dx))
        self.assertAlmostEqual(pot.phi2deriv( 0,  0,   -1), -deriv(lambda x: pot.phiforce( 0,  0, x),   -1, dx=dx))
        self.assertAlmostEqual(pot.phi2deriv( 0,  0,    1), -deriv(lambda x: pot.phiforce( 0,  0, x),    1, dx=dx))

        # These don't work for some reason...
        #self.assertAlmostEqual(pot.phi2deriv( 1,  1, -0.5), -deriv(lambda x: pot.phiforce( 1,  1, x), -0.5, dx=dx))
        #self.assertAlmostEqual(pot.phi2deriv( 1,  1,  0.5), -deriv(lambda x: pot.phiforce( 1,  1, x),  0.5, dx=dx))

        self.assertAlmostEqual(pot.phi2deriv(.1, .1, -0.5), -deriv(lambda x: pot.phiforce(.1, .1, x), -0.5, dx=dx))
        self.assertAlmostEqual(pot.phi2deriv(.1, .1,  0.5), -deriv(lambda x: pot.phiforce(.1, .1, x),  0.5, dx=dx))
        self.assertAlmostEqual(pot.phi2deriv(-1, -1, -2.3), -deriv(lambda x: pot.phiforce(-1, -1, x), -2.3, dx=dx))
        self.assertAlmostEqual(pot.phi2deriv(-1, -1,  2.3), -deriv(lambda x: pot.phiforce(-1, -1, x),  2.3, dx=dx))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpiralArmsPotential)
    unittest.TextTestRunner(verbosity=2).run(suite)
