import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from galpy.potential import SpiralArmsPotential
from galpy.util import bovy_coords


ts = np.linspace(0, 1, 60)
sp = SpiralArmsPotential(omega=2 * np.pi)

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

n = 50
xmin = -2
xmax = 2
ymin = -2
ymax = 2
xs = np.linspace(xmin, xmax, n)
ys = np.linspace(ymin, ymax, n)

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

ax1.set_title('Potential')
ax2.set_title('Density')
ax3.set_title('Rforce')
ax4.set_title('phiforce')

pot = np.zeros((n, n))
dens = np.zeros((n, n))
Rforce = np.zeros((n, n))
phiforce = np.zeros((n, n))

pot_ims = []
dens_ims = []
Rforce_ims = []
phiforce_ims = []


def plot(t):
    for ii in range(n):
        for jj in range(n):
            R, phi, z = bovy_coords.rect_to_cyl(xs[ii], ys[jj], 0)
            pot[ii, jj] = sp(R, z, phi, t)
            dens[ii, jj] = sp.dens(R, z, phi, t)
            Rforce[ii, jj] = sp.Rforce(R, z, phi, t)
            phiforce[ii, jj] = sp.phiforce(R, z, phi, t)

    return [ax1.imshow(pot, cmap='coolwarm'),
            ax2.imshow(dens, cmap='coolwarm'),
            ax3.imshow(Rforce, cmap='coolwarm'),
            ax4.imshow(phiforce, cmap='coolwarm')]


for t in ts:
    plts = plot(t)

    pot_ims.append([plts[0]])
    dens_ims.append([plts[1]])
    Rforce_ims.append([plts[2]])
    phiforce_ims.append([plts[3]])
    # im = sp.plot(t=t, xy=True, rmin=-2, rmax=2, zmin=-2, zmax=2, ncontours=3, nrs=100)
    # im.set_cmap('coolwarm')
    # im.colorbar = plt.colorbar(im)
    # im.__getattribute__('axes').set_ylim([2, -2])
    # ims.append((im,))


def main():
    ani1 = ArtistAnimation(fig, pot_ims, blit=False, interval=100, repeat=True)
    ani2 = ArtistAnimation(fig, dens_ims, blit=False, interval=100, repeat=True)
    ani3 = ArtistAnimation(fig, Rforce_ims, blit=False, interval=100, repeat=True)
    ani4 = ArtistAnimation(fig, phiforce_ims, blit=False, interval=100, repeat=True)

    # ani1.save('SpiralArmsPotential_potential_animation.mp4')
    # ani2.save('SpiralArmsPotential_density_animation.mp4')
    # ani3.save('SpiralArmsPotential_Rforce_animation.mp4')
    # ani4.save('SpiralArmsPotential_phiforce_animation.mp4')

    plt.show()

if __name__ == '__main__':
    main()
