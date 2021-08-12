# -*- coding: utf-8 -*-

"""
Module et_md3.verletlist.vlbuilders
===================================

A submodule for Verlet list builders
"""
import numpy as np


def build_simple(vl, r, keep2d=False):
    """Build the Verlet list from the positions.

    Brute force approach, in the simplest way.
    This algorithm has complexity O(N).

	:param vl: the Verlet list object to build.
	:param numpy.ndarray: Numpy array with atom position coordinates, shape = (n,3).
	:param bool keep2d: (For debugging purposes). Keep the (internal) 2D data structure.
    """
    vl.allocate_2d_(r.shape[0])
    rc2 = vl.cutoff ** 2
    for i in range(vl.natoms - 1):
        ri = r[i, :]
        for j in range(i + 1, vl.natoms):
            rj = r[j, :]
            rij = rj - ri
            rij2 = np.dot(rij, rij)
            if rij2 <= rc2:
                vl.add(i, j)
    vl.linearise(keep2d=keep2d)
