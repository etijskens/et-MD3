"""Tests for `et_md3.atoms` submodule."""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(0,'.')

import pytest
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os

no_show = True
no_show = False
if not no_show:
    no_show = "NO_SHOW" in os.environ
print(f'no_show={no_show}')

import et_md3.atoms as atoms

def assert_n_dtype(a, n, dtype):
    for ar in a.arrays:
        assert len(ar) == n
        assert ar.dtype == dtype

def test_atoms_default():
    n = 10
    dtype = float
    a = atoms.Atoms(n)
    assert_n_dtype(a,n,dtype)


def test_atoms_float():
    n = 10
    dtype = float
    a = atoms.Atoms(n, dtype=dtype)
    assert_n_dtype(a,n,dtype)


def test_atoms_single():
    n = 10
    dtype = np.single
    a = atoms.Atoms(n, dtype=dtype)
    assert_n_dtype(a,n,dtype)


def test_atoms_double():
    n = 10
    a = atoms.Atoms(n)
    dtype = np.double
    assert_n_dtype(a,n,dtype)


def test_atoms_bad_dtype():
    n = 10
    dtype = int
    with pytest.raises(TypeError):
        a = atoms.Atoms(n, dtype=dtype)

def test_random_positions():
    n = 10000
    dtype = np.double
    a = atoms.Atoms(n)

    lc = (1.,1.,1.)
    uc = (2.,2.,2.)
    a.random_positions(lc, uc)

    for ia in range(n):
        for i in range(3):
            assert np.all(lc <= a.r[ia,i])
            assert np.all(      a.r[ia,i] < uc)


def test_random_positions_single():
    n = 10000
    dtype = np.single
    a = atoms.Atoms(n,dtype=dtype)

    lc = (1.,1.,1.)
    uc = (2.,2.,2.)
    a.random_positions(lc, uc)

    for ia in range(n):
        for i in range(3):
            assert lc[i] <= a.r[ia,i] < uc[i]

    assert a.r.dtype == np.single


def test_resize():
    n = 5
    dtype = np.single
    a = atoms.Atoms(n,dtype=dtype)
    for ar in a.arrays:
        d = len(ar.shape)
        if d == 1:
            x = 0.0
        elif d == 2:
            x = np.zeros(3, dtype=ar.dtype)
        for iar in range(n):
            ar[iar] = x
            x += 1
        print()
    x = 0.0
    for ar in a.arrays:
        if d == 1:
            x = 0.0
        elif d == 2:
            x = np.zeros(3, dtype=ar.dtype)
        for iar in range(n):
            assert np.all(ar[iar] == x)
            x += 1

    # Double the size
    n2 = 10
    a.resize(n2)
    assert_n_dtype(a,n2,dtype)
    for ar in a.arrays:
        if d == 1:
            x = 0.0
        elif d == 2:
            x = np.zeros(3, dtype=ar.dtype)
        for iar in range(n):
            assert np.all(ar[iar] == x)
            x += 1

    # halve the size
    a.resize(n)
    assert_n_dtype(a,n,dtype)
    for ar in a.arrays:
        if d == 1:
            x = 0.0
        elif d == 2:
            x = np.zeros(3, dtype=ar.dtype)
        for iar in range(n):
            assert np.all(ar[iar] == x)
            x += 1


def test_lattice_positions():
    n = 5
    dtype = np.single
    a = atoms.Atoms(n,dtype=dtype)
    a.lattice_positions()
    zero = np.zeros(3, dtype)
    assert a.n == 4
    ia = 0
    assert np.all(a.r[ia] == zero)
    ia = 1
    assert np.all(a.r[ia] == np.array([0.5, 0.5, 0.0], dtype))
    ia = 2
    assert np.all(a.r[ia] == np.array([0.5, 0.0, 0.5], dtype))
    ia = 3
    assert np.all(a.r[ia] == np.array([0.0, 0.5, 0.5], dtype))


def test_lattice_positions_2():
    n = 8
    dtype = np.single
    a = atoms.Atoms(n,dtype=dtype)
    a.lattice_positions(upper_corner=(2,1,1))
    assert a.n == 8
    zero = np.zeros(3, dtype)
    ia = 0
    assert np.all(a.r[ia] == zero)
    ia += 1
    assert np.all(a.r[ia] == np.array([0.5, 0.5, 0.0], dtype))
    ia += 1
    assert np.all(a.r[ia] == np.array([0.5, 0.0, 0.5], dtype))
    ia += 1
    assert np.all(a.r[ia] == np.array([0.0, 0.5, 0.5], dtype))
    ia += 1
    assert np.all(a.r[ia] == np.array([1., 0.0, 0.0], dtype))
    ia += 1
    assert np.all(a.r[ia] == np.array([1.5, 0.5, 0.0], dtype))
    ia += 1
    assert np.all(a.r[ia] == np.array([1.5, 0.0, 0.5], dtype))
    ia += 1
    assert np.all(a.r[ia] == np.array([1.0, 0.5, 0.5], dtype))
    ia += 1


def test_plot_8():
    """for visual inspection"""
    n = 8
    dtype = np.single
    a = atoms.Atoms(n,dtype=dtype)
    a.lattice_positions(upper_corner=(2,1,1))
    a.plot()
    if not no_show:
        plt.show()


def test_add_noise():
    n = 80
    dtype = np.single
    upper_corner = (20, 1, 1)
    a = atoms.Atoms(n, dtype=dtype)
    b = atoms.Atoms(n, dtype=dtype)
    a.lattice_positions(upper_corner=upper_corner)
    b.lattice_positions(upper_corner=upper_corner)
    r = 0.05
    a.add_noise(r)
    abr = a.r - b.r
    d2 = np.dot(abr,abr.transpose())
    rr = r**2
    # for ia in range(len(d2)):
    #     print(ia, d2[ia], rr, d2[ia]<rr)
    #     assert d2[ia] <= rr
    assert np.all(d2 < rr)
    assert_n_dtype(a,n,dtype)


def test_apply_PBC():
    n = 8
    dtype = np.single
    upper_corner = (2, 1, 1)
    a = atoms.Atoms(n, dtype=dtype)
    a.lattice_positions(upper_corner=upper_corner)
    r = 0.5
    a.add_noise(r)
    print(a.apply_PBC(collect=True))
    # apply PBC again, now not a single atom should have moved
    moved = a.apply_PBC(collect=True)
    assert not moved


def test_scale_forces():
    n = 5
    atms = atoms.Atoms(n)
    for i in range(n):
        atms.a[i,:] = 1.0 + i
    atms.m = 2.0 * np.ones_like(atms.a)
    atms.scale_forces(impl='py')
    for i in range(n):
        assert np,all(atms.a[i,:] == (1.0 + i)/2.0)
    atms.m = 2.0
    atms.scale_forces(impl='py')
    for i in range(n):
        assert np,all(atms.a[i,:] == (1.0 + i)/4.0)



# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_atoms_default

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
