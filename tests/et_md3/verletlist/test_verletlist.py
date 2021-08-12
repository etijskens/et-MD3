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


# def test_build_1():
#     """Verify VerletList.build against VerletList.build_simple."""
#
#     cutoff = 2.5
#     x = np.array([0.0, 1, 2, 3, 4])
#     n_atoms = len(x)
#     r = np.empty((n_atoms,3))
#     msg = ['x00','0x0','00x']
#     for ir in range(3):
#         r[:,:] = 0.0
#         r[:,ir] = x
#         print(f'*** {msg[ir]} ***')
#
#         VerletList = et_md3.verletlist.implementation(impl='py')
#         vl = VerletList(cutoff=cutoff)
#         vl.build(r)
#         print(vl)
#         pairs = et_md3.verletlist.vl2set(vl)
#
#         vlsimple = VerletList(cutoff=cutoff)
#         vlsimple.build_simple(r)
#         print(vlsimple)
#         expected = et_md3.verletlist.vl2set(vlsimple)
#         assert pairs == expected
#         assert np.all(vl.vl_size == np.array([2,2,2,1,0]))
#         assert np.all(vl.vl_list == np.array([1,2,2,3,3,4,4]))
#
# def test_build_2():
#     """Verify VerletList.build against VerletList.build_simple."""
#     cutoff = 2.0
#     atoms = Atoms()
#     atoms.lattice_positions(upper_corner=(5,5,5))
#
#     VerletList = et_md3.verletlist.implementation(impl='py')
#     vl = VerletList(cutoff=cutoff)
#     vl.build(atoms.r)
#     print(vl)
#     pairs = et_md3.verletlist.vl2set(vl)
#
#     vlsimple = VerletList(cutoff=cutoff)
#     vlsimple.build_simple(atoms.r)
#     print(vlsimple)
#     pairs_simple = et_md3.verletlist.vl2set(vlsimple)
#     assert pairs == pairs_simple

# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_neighbours

    print(f'__main__ running {the_test_you_want_to_debug}')
    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
