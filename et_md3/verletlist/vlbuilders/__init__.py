# -*- coding: utf-8 -*-

"""
Module et_md3.verletlist.vlbuilders
===================================

A submodule for Verlet list builders
"""
import numpy as np

import et_md3.verletlist


def build_simple(vl, r, keep2d=False):
    """Build a Verlet list from atom positions.

    Brute force approach, in the simplest way.
    This algorithm has complexity O(N).

    :param vl: the Verlet list object to build.
    :param numpy.ndarray: Numpy array with atom position coordinates, shape = (n,3).
    :param bool keep2d: (For debugging purposes). Keep the (internal) 2D data structure.
    """
    vl.allocate_2d(r.shape[0])
    rc2 = vl.cutoff() ** 2
    natoms = vl.natoms()
    for i in range(natoms - 1):
        ri = r[i, :]
        for j in range(i + 1, natoms):
            rj = r[j, :]
            rij = rj - ri
            rij2 = np.dot(rij, rij)
            if rij2 <= rc2:
                vl.add(i, j)
    vl.linearise(keep2d)


def build(vl, r, keep2d=False):
    """Build a Verlet list from atom positions.

    Brute force approach, but using array arithmetic, rather than pairwise
    computations. This algorithm has complexity O(N), but is significantly
    faster than ``build_simple()``.

    :param vl: the Verlet list object to build.
    :param np.ndarray r: list of numpy arrays with atom coordinates: r = [x, y, z]
    :param bool keep2d: if True the 2D Verlet list data structure is not deleted after linearisation.
    """
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    vl.allocate_2d(len(x))
    rc2 = vl.cutoff() ** 2

    natoms = vl.natoms()
    ri2 = np.empty((natoms,), dtype=r.dtype)
    rij = np.empty_like(r)
    for i in range(natoms - 1):
        rij[i + 1:, :] = r[i + 1:, :] - r[i, :]

        if isinstance(vl, et_md3.verletlist.VL):
            if vl.debug:
                ri2 = 0
        ri2[i + 1:] = np.einsum('ij,ij->i', rij[i + 1:, :], rij[i + 1:, :])
        for j in range(i + 1, natoms):
            if ri2[j] <= rc2:
                vl.add(i, j)
    vl.linearise(keep2d)


def build_grid(vl, r, grid, keep2d=False):
    """Build Verlet lists using a grid.

    This algorithm has complexity O(N).

    :param list r: list of numpy arrays with atom coordinates: r = [x, y, z]
   """
    x = r[0]
    y = r[1]
    z = r[2]
    if not grid.linearised():
        raise ValueError("The grid list must be built and linearised first.")

    self.allocate_2d(len(x))
    rc2 = self.cutoff ** 2
    # loop over all cells
    for m in range(grid.K[2]):
        for l in range(grid.K[1]):
            for k in range(grid.K[0]):
                cklm = grid.cell_list(k,l,m)
                natoms_in_cklm = len(cklm)
                # loop over all atom pairs in cklm
                for ia in range(natoms_in_cklm):
                    i = cklm[ia]
                    for j in cklm[ia + 1:]:
                        rij2 = (x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2
                        if rij2 <= rc2:
                            self.add(i, j)
                # loop over neighbouring cells. If the cell does not exist an IndexError is raised
                for klm2 in ( (k+1,l  ,m)   # one ahead in the x-direction
                            , (k-1,l+1,m)   # three ahead in the y-direction
                            , (k  ,l+1,m)
                            , (k+1,l+1,m)
                            , (k-1,l-1,m+1) # nine ahead in the z-direction
                            , (k  ,l-1,m+1)
                            , (k+1,l-1,m+1)
                            , (k-1,l  ,m+1)
                            , (k  ,l  ,m+1)
                            , (k+1,l  ,m+1)
                            , (k-1,l+1,m+1)
                            , (k  ,l+1,m+1)
                            , (k+1,l+1,m+1)
                            ):
                    try:
                        cklm2 = grid.cell_list(*klm2)
                    except IndexError:
                        pass  # Cell kl2 does not exist
                    else:  # The else clause is executed only when the try clause does not raise an error
                        # loop over all atom pairs i,j with i in cklm and j in cklm2
                        for i in cklm:
                            for j in cklm2:
                                rij2 = (x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2
                                if rij2 <= rc2:
                                    self.add(i, j)

    self.linearise(keep2d=keep2d)
