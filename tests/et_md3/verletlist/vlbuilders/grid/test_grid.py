#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et_md3.verletlist.vlbuilders.grid` package."""
import sys
sys.path.insert(0,'.')

# import pytest
import numpy as np
from et_md3.atoms import Atoms
from et_md3.verletlist.vlbuilders.grid import Grid, row_major_index

def test_row_major_index_2d():
    K = (2,3)
    l_expected = 0
    for i in range(K[0]):
        for j in range(K[1]):
            l = row_major_index((i,j), K)
            assert l == l_expected
            l_expected += 1


def test_row_major_index_3d():
    K = (2,3,4)
    l_expected = 0
    for i in range(K[0]):
        for j in range(K[1]):
            for k in range(K[2]):
                l = row_major_index((i,j,k), K)
                assert l == l_expected
                l_expected += 1


def test_build():
    a = 0.5
    atoms = Atoms(n=1)
    atoms.lower_corner = np.zeros(3 ,dtype=float)
    atoms.upper_corner = 2*np.ones(3, dtype=float)
    atoms.r[0,:] = a
    grid = Grid(cell_size=1.0, atoms=atoms)
    grid.clear()
    for d in range(3):
        assert grid.cl.shape[d] == 2 # number of cells in d-direction

    grid.build(linearise=False)

    for k in range(2):
        for l in range(2):
            for m in range(2):
                if k==0 and l==0 and m==0:
                    assert grid.cl[k, l, m, 0] == 1 # one atom in this cell
                    assert grid.cl[k, l, m, 1] == 0 # atom #0
                else:
                    assert grid.cl[k,l,m,0] == 0    # no atoms in this cell
    grid.linearise()
    n_cells = grid.cl_size.shape[0]
    assert n_cells == 2**3
    for rmi in range(n_cells):
        if rmi == 0:
            assert grid.cl_size[rmi] == 1                   # one atom in this cell
            assert grid.cl_list[grid.cl_offset[rmi]] == 0   # atom #0
        else:
            assert grid.cl_size[rmi] == 0                   # no atoms in this cell
    grid.clear()


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_build

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
