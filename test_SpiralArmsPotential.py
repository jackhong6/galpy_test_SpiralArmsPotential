import galpy.potential_src.SpiralArmsPotential as spiral
import matplotlib.pyplot as plt
import numpy as np

spiral_pot = spiral.SpiralArmsPotential()

spiral_pot.plot(xy=True, rmin=-1, rmax=1, zmin=-0.99, zmax=0.99)


plt.show()


