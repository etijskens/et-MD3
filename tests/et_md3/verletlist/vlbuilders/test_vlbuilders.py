#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for sub-module et_md3.verletlist.vlbuilders."""

import sys
sys.path.insert(0,'.')

import numpy as np
import pytest


import et_md3.verletlist
import et_md3.verletlist.vlbuilders

def test_build_simple_1():
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
        vl = VerletList(cutoff=2.5)
        et_md3.verletlist.vlbuilders.build_simple(vl, r, keep2d=True)
        print(vl)
        assert     vl.has((0, 1))
        assert     vl.has((0, 2))
        assert not vl.has((0, 3))
        assert not vl.has((0, 4))
        assert not vl.has((1, 0))
        assert     vl.has((1, 2))
        assert     vl.has((1, 3))
        assert not vl.has((1, 4))
        assert not vl.has((2, 0))
        assert not vl.has((2, 1))
        assert     vl.has((2, 3))
        assert     vl.has((2, 4))
        assert not vl.has((3, 0))
        assert not vl.has((3, 1))
        assert not vl.has((3, 2))
        assert     vl.has((3, 4))
        assert not vl.has((4, 0))
        assert not vl.has((4, 1))
        assert not vl.has((4, 2))
        assert not vl.has((4, 3))
        assert np.all(vl.vl_size == np.array([2,2,2,1,0]))
        assert np.all(vl.vl_list == np.array([1,2,2,3,3,4,4]))


def test_build_simple_2():
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
        et_md3.verletlist.vlbuilders.build_simple(vl, r, keep2d=True)
        print(vl)
        assert     vl.has((0, 1))
        assert     vl.has((0, 2))
        assert not vl.has((0, 3))
        assert not vl.has((0, 4))
        assert not vl.has((1, 0))
        assert     vl.has((1, 2))
        assert     vl.has((1, 3))
        assert not vl.has((1, 4))
        assert not vl.has((2, 0))
        assert not vl.has((2, 1))
        assert     vl.has((2, 3))
        assert     vl.has((2, 4))
        assert not vl.has((3, 0))
        assert not vl.has((3, 1))
        assert not vl.has((3, 2))
        assert     vl.has((3, 4))
        assert not vl.has((4, 0))
        assert not vl.has((4, 1))
        assert not vl.has((4, 2))
        assert not vl.has((4, 3))
        assert np.all(vl.vl_size == np.array([2,2,2,1,0]))
        assert np.all(vl.vl_list == np.array([1,2,2,3,3,4,4]))


def test_build_simple_2b():
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
        et_md3.verletlist.vlbuilders.build_simple(vl, r, keep2d=True)
        # print(vl)
        pairs = et_md3.verletlist.vl2set(vl)
        expected = {(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)}
        assert pairs == expected


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_build_simple_2b

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
