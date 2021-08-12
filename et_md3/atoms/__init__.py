# -*- coding: utf-8 -*-

"""
Module et_md3.atoms
===================

A submodule for representing atoms

"""

import numpy as np
import matplotlib.pyplot as plt


class Atoms:
    """Class for a collection of atoms.


    :param int n: number of atoms to generate. All atom array are created with np.empty,
        and they are not initialized.
    :param float|np.single dtype: type of the atom arrays, float (np.double) by default.
    """
    def __init__(self, n=10, dtype=float, m=None, zero=False):
        accepted_dtypes = (float,np.single)
        if dtype in accepted_dtypes:
            self.dtype = dtype
        else:
            raise TypeError(f"Expecting `dtype` to be in {accepted_dtypes}, got `{dtype}`.")

        if zero:
            self.r = np.zeros((n,3), dtype=dtype)
            self.v = np.zeros((n,3), dtype=dtype)
            self.a = np.zeros((n,3), dtype=dtype)
            if m is None:
                self.m = np.ones (n, dtype=dtype)
            else:
                self.m = m
        else:
            self.r = np.empty((n,3), dtype=dtype)
            self.v = np.empty((n,3), dtype=dtype)
            self.a = np.empty((n,3), dtype=dtype)
            if m is None:
                self.m = np.empty( n   , dtype=dtype)
            else:
                self.m = m

        self.arrays = [self.r, self.v, self.a, self.m]

        self.lower_corner = None
        self.upper_corner = None


    @property
    def n(self):
        """Return the number of atoms"""
        return self.r.shape[0]


    def resize(self, n, positions_only=False):
        """resize all atom arrays """
        arrays = [self.r] if positions_only else self.arrays
        for ar in arrays:
            newshape = list(ar.shape)
            newshape[0] = n
            ar.resize(newshape, refcheck=False)


    def random_positions(self, lower_corner=(0,0,0), upper_corner=(1,1,1)):
        """fill the positions with random values inside a box."""
        for i in range(3):
            self.r[:,] = lower_corner[i] + (upper_corner[i] - lower_corner[i])*np.random.random(self.r.shape)
        self.lower_corner = np.array(lower_corner)
        self.upper_corner = np.array(upper_corner)


    def lattice_positions(self, lower_corner=(0,0,0), upper_corner=(1,1,1), r0=None, cell='fcc'):
        """Fill the box with lattice points."""
        if cell == 'fcc':
            unitcell = np.array([[0.0,0.0,0.0]
                                ,[0.5,0.5,0.0]
                                ,[0.5,0.0,0.5]
                                ,[0.0,0.5,0.5]])
            if not r0:
                r0 = np.sqrt(0.5)
                a = 1.0
            else:
                a = np.sqrt(2) * r0

        elif cell == 'bcc':
            unitcell = np.array([[0.0,0.0,0.0]
                                ,[0.5,0.5,0.5]])
            if not r0:
                r0 = np.sqrt(3.0)/2.0
                a = 1.0
            else:
                a = 2*r0/np.sqrt(3.0)

        elif cell == 'primitive':
            unitcell = np.array([[0.0,0.0,0.0]])
            if not r0:
                r0 = 0.5
                a = 1.0
            else:
                a = r0

        else:
                raise NotImplementedError(f"Unknown unit cell `{cell}`.")

        unitcell *= a
        Z = unitcell.shape[0]

        # estimate the number of unit cells that fit in the box, to be able to allocate
        # the arrays
        lc = np.array(lower_corner)
        uc = np.array(upper_corner)
        w = uc - lc
        n = np.ceil(w/a).astype(int)
        N = int(Z * n[0] * n[1] * n[2])
        self.resize(N, positions_only=True) # this is an upper bound
        ijk = np.array([0.0, 0.0, 0.0])
        ia = 0
        for k in range(n[2]):
            ijk[2] = k
            for j in range(n[1]):
                ijk[1] = j
                for i in range(n[0]):
                    ijk[0] = i
                    p = ijk + unitcell
                    for iz in range(Z):
                        if np.all(p[iz,:] < w) :
                            self.r[ia,:] = p[iz]
                            ia += 1

        self.resize(ia) # this is the real size.

        # add the offset (lower_corner)
        if not np.all(lc == 0.0):
            self.r += lc

        # Store the box
        self.lower_corner = lc
        self.upper_corner = uc


    def add_noise(self, d):
        """Displace all atoms in a random direction by a random distance in [0,d[."""
        n = self.n
        r = np.random.rand(n)*d 				# noise magnitude
        alpha = np.random.rand(n)*(2*np.pi)     # angle in horizontal plane
        theta = (np.random.rand(n) - 0.5)*np.pi # angle in vertical plane
        costheta = np.cos(theta)

        self.r[:,0] += r*np.cos(alpha)*costheta
        self.r[:,1] += r*np.sin(alpha)*costheta
        self.r[:,2] += r*np.sin(theta)


    def apply_PBC(self, collect=False):
        """Apply periodic boundary conditions.

        :param bool collect: return set of moved atoms
        :return: if collect is True, returns the set of moved atoms, otherwise returns None.
        """
        if not self.lower_corner is None and not self.upper_corner is None:
            moved  = set()
            for id in range(3):
                ar = self.r[id]
                lc = self.lower_corner[id]
                uc = self.upper_corner[id]
                d = uc - lc
                below = np.argwhere(ar < lc)
                if len(below):
                    ar[below] += d
                    if collect:
                        for atom in below:
                            moved.add(atom[0])
                above = np.argwhere(ar >= uc)
                if len(above):
                    ar[above] -= d
                    if collect:
                        for atom in above:
                            moved.add(atom[0])

        if collect:
            return moved


    def plot(self, box=True, atoms=True):
        """Plot the box and the atoms"""

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # Setting the axes properties
        ax.set_title('atoms')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_aspect('equal') # not supported for 3D.


        if box and not self.lower_corner is None and not self.upper_corner is None:
            x0 = self.lower_corner[0]
            y0 = self.lower_corner[1]
            z0 = self.lower_corner[2]
            x1 = self.upper_corner[0]
            y1 = self.upper_corner[1]
            z1 = self.upper_corner[2]
            x = [x0, x0, x0, x0]
            y = [y1, y0, y0, y1]
            z = [z1, z1, z0, z0]
            plt.plot(x,y,z)
            x = [x0, x1, x1, x0]
            y = [y0, y0, y0, y0]
            z = [z0, z0, z1, z1]
            plt.plot(x, y, z)
            x = [x1, x1, x1, x1]
            y = [y0, y1, y1, y0]
            z = [z0, z0, z1, z1]
            plt.plot(x, y, z)
            x = [x1, x0, x0, x1]
            y = [y1, y1, y1, y1]
            z = [z1, z1, z0, z0]
            plt.plot(x, y, z)

        if atoms:
            plt.plot(self.r[:,0], self.r[:,1], self.r[:,2], '.')


    def print(self):
        """print the positions of the atoms, mainly for debugging"""
        for i in range(self.n):
            print(f'i={i}, r={self.r[i,:]}')


    def scale_forces(self):
        """Convert the forces (as computed by the potential) to accelerations.

        F = ma, a currently contains the forces, so we must divide by m to obtain the
        accelerations.
        """
        # Note: np.double and float are not the same type.
        if isinstance(self.m, (float, np.double, np.single)): # all atoms have the same mass.
            self.a /= self.m

        elif self.m.shape[0] == self.n: # atoms have different masses.
            # todo: efficiency issue: python loop
            for i in range(self.n):
                self.a[i, :] /= self.m[i]

        else:
            raise ValueError(f'atoms.m has wrong dimension ({self.m.shape[0]} instead of {self.n}).')
