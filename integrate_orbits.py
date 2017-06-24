from __future__ import division
from galpy.orbit import Orbit
from galpy.potential import SpiralArmsPotential, MWPotential2014
import numpy as np
import matplotlib.pyplot as plt

sp = SpiralArmsPotential(amp=1)  # amp <= 3 for positive density
mp = MWPotential2014
pot = [sp] + mp

orb = Orbit(vxvv=[1, 0.1, 1.1, 0, 0.1, 0])
orb2 = Orbit(vxvv=[1, 0.1, 1.1, 0, 0.1, 0])

ts = np.linspace(0,100,1000)
orb.integrate(ts, pot, method='dopr54_c')
orb2.integrate(ts, pot, method='odeint')

orb.plot(d1='x', d2='y')
orb.plot()

orb2.plot(d1='x', d2 = 'y')
orb2.plot()
#orb.plotE()
plt.show()