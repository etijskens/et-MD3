#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for sub-module et_md3.verletlist."""

# import pytest
import sys
sys.path.insert(0,'.')


from et_md3.atoms import Atoms
import et_md3.verletlist
from et_md3.verletlist.vlbuilders import build_simple
# from et_md3.grid import Grid

import numpy as np



def test_vl2pairs():
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    r = np.empty((n_atoms,3))
    msg = ['x00','0x0','00x']
    for ir in range(3):
        r[:,:] = 0.0
        r[:,ir] = x
        print(f'*** {msg[ir]} ***')

        VerletList = et_md3.verletlist.implementation(impl='py')
        vl = VerletList(cutoff=2.0)
        build_simple(vl, r, keep2d=True)
        print(vl)
        print(et_md3.verletlist.vl2set(vl))


def test_neighbours():
    """"""
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    r = np.empty((n_atoms,3))
    msg = ['x00','0x0','00x']
    for ir in range(3):
        r[:,:] = 0.0
        r[:,ir] = x
        print(f'*** {msg[ir]} ***')

        VerletList = et_md3.verletlist.implementation(impl='py')
        vl = VerletList(cutoff=2.0)
        build_simple(vl, r)
        print(vl.vl_size[0])
        assert vl.vl_size[0] == 2
        vl0 = vl.verlet_list(0)
        print(vl0)
        assert vl0[0] == 1
        assert vl0[1] == 2


# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_neighbours

    print(f'__main__ running {the_test_you_want_to_debug}')
    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
