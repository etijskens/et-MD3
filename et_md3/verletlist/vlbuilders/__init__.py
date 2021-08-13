# -*- coding: utf-8 -*-

"""
Module et_md3.verletlist.vlbuilders
===================================

A submodule for Verlet list builders
"""
import numpy as np


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
        if vl.debug:
            ri2 = 0
        ri2[i + 1:] = np.einsum('ij,ij->i', rij[i + 1:, :], rij[i + 1:, :])
        for j in range(i + 1, natoms):
            if ri2[j] <= rc2:
                vl.add(i, j)
    vl.linearise(keep2d)
