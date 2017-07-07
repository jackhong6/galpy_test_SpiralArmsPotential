import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from galpy.potential import SpiralArmsPotential
from galpy.util import bovy_coords


ts = np.linspace(0, 1, 100)
sp = SpiralArmsPotential(omega=2 * np.pi)

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

n = 250
xmin = -2
xmax = 2
ymin = -2
ymax = 2
xs = np.linspace(xmin, xmax, n)
ys = np.linspace(ymin, ymax, n)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)

pot = np.zeros((n, n))

for ii in range(n):
    for jj in range(n):
        R, phi, z = bovy_coords.rect_to_cyl(xs[ii], ys[jj], 0)
        pot[ii, jj] = sp(R, z, phi, 0)

im0 = ax1.imshow(pot.T, cmap='coolwarm', origin='lower')
fig.colorbar(im0, ax=ax1, fraction=0.046, pad=0.04)

ax1.set_title('Spiral Arms Potential')


pot_ims = []

def plot(t):
    for ii in range(n):
        for jj in range(n):
            R, phi, z = bovy_coords.rect_to_cyl(xs[ii], ys[jj], 0)
            pot[ii, jj] = sp(R, z, phi, t)
            #pot[ii, jj] = jj

    im1 = ax1.imshow(pot.T, cmap='coolwarm', origin='lower')

    return [im1]


for t in ts:
    plts = plot(t)

    pot_ims.append([plts[0]])
    # im = sp.plot(t=t, xy=True, rmin=-2, rmax=2, zmin=-2, zmax=2, ncontours=3, nrs=100)
    # im.set_cmap('coolwarm')
    # im.colorbar = plt.colorbar(im)
    # im.__getattribute__('axes').set_ylim([2, -2])
    # ims.append((im,))


def main():
    ani1 = ArtistAnimation(fig, pot_ims, blit=False, interval=100, repeat=True)

    ani1.save('SpiralArmsPotential_potential_animation.mp4')

    plt.show()

if __name__ == '__main__':
    main()
