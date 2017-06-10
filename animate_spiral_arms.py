import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from galpy.potential import SpiralArmsPotential

ts = np.linspace(0, 1, 10)
sp = SpiralArmsPotential(omega=2 * np.pi)

im0 = sp.plot(t=ts[0], xy=True, rmin=-2, rmax=2, zmin=-2, zmax=2, ncontours=7)
im0.set_cmap('coolwarm')
im0.colorbar = plt.colorbar(im0)
im0.__getattribute__('axes').set_ylim([2, -2])

fig = plt.figure()

ims = [(im0,)]

im0.__getattribute__('figure').clf()
np.delete(ts, 0)

for t in ts:
    im = sp.plot(t=t, xy=True, rmin=-2, rmax=2, zmin=-2, zmax=2, ncontours=7)
    im.set_cmap('coolwarm')
    im.colorbar = plt.colorbar(im)
    im.__getattribute__('axes').set_ylim([2, -2])
    ims.append((im,))
    im.__getattribute__('figure').close()


def main():
    ani = ArtistAnimation(fig, ims, blit=False, interval=50, repeat=True)
    ani.save('test.mp4')

if __name__ == '__main__':
    main()
