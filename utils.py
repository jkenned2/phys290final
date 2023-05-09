# useful functions are kept here
import numpy as np


# computes acceleration for all particles
def getAccel(pos, mass, G, softening):
    # get positions
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # get pairwise separations
    rx = x.T - x
    ry = y.T - y
    rz = z.T - z

    # get 1/r^3
    inv_r3 = (rx**2 + ry**2 + rz**2 + softening**2)**(-1.5)

    # get acceleration
    ax = G * (rx * inv_r3) @ mass
    ay = G * (ry * inv_r3) @ mass
    az = G * (rz * inv_r3) @ mass
    return np.hstack((ax, ay, az))


def getEnergy(pos, vel, mass, G):
    # get kinetic energy
    KE = 0.5 * np.sum(np.sum(mass * vel**2))

    # get positions
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # get pairwise separations
    rx = x.T - x
    ry = y.T - y
    rz = z.T - z

    # get 1/r
    inv_r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle so interactions are counted only once
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE
