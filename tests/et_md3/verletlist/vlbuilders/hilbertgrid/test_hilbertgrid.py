#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et-MD` package."""

import sys
sys.path.insert(0,'.')

import numpy as np
import pytest

import et_md3.verletlist.vlbuilders.hilbertgrid
import et_md3.verletlist.vlbuilders.hilbertgrid.spatialsorting
from et_md3.atoms import Atoms

HilbertGrid = et_md3.verletlist.vlbuilders.hilbertgrid.HilbertGrid
build_hl = {'py' : et_md3.verletlist.vlbuilders.hilbertgrid.build_hl
           ,'cpp': et_md3.verletlist.vlbuilders.hilbertgrid.spatialsorting.build_hl
           }


def test_hilbert_cube():
    for k in (0,1,2):
        k2 = et_md3.verletlist.vlbuilders.hilbertgrid.hilbert_cube(k)
        assert k2 == k

    expected = 2
    n = 6
    for i in range(n): # test all cases up to 2**(n+1)
        k0 = expected + 1
        expected *= 2
        for k in range(k0,expected+1):
            k2 = et_md3.verletlist.vlbuilders.hilbertgrid.hilbert_cube(k)
            print(f'{k} -> {k2}, expected = {expected}')
            assert k2 == expected


def test_build_hl_1():
    h = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7], dtype=HilbertGrid.H_t)
    ncells = 8
    hl_offset, hl_natoms = build_hl['py'](h)
    assert np.all(hl_offset == [0, 2, 4, 6, 8,10,12,14])
    assert np.all(hl_natoms == [2, 2, 2, 2, 2, 2, 2, 2])

def test_build_hl_2():
    h = np.array([0,0,2,2,3,3,4,4,5,5,6,6,7,7], dtype=HilbertGrid.H_t)
    ncells = 8
    hl_offset, hl_natoms = build_hl['py'](h)
    assert np.all(hl_offset == [0, 2, 2, 4, 6, 8,10,12])
    assert np.all(hl_natoms == [2, 0, 2, 2, 2, 2, 2, 2])

def test_build_hl_3():
    h = np.array([2,2,3,3,4,4,5,5,6,6,7,7], dtype=HilbertGrid.H_t)
    ncells = 8
    hl_offset, hl_natoms = build_hl['py'](h)
    assert np.all(hl_offset == [0, 0, 0, 2, 4, 6, 8,10])
    assert np.all(hl_natoms == [0, 0, 2, 2, 2, 2, 2, 2])

def test_HilbertGrid_ideal_boxes():
    """test with ideal boxes: cubic, and the number of cells in each direction is a power of 2."""
    for cell in ('primitive','bcc','fcc'):
        # for cutoff in (1.0,2.0):
        cutoff =1
        for upper_corner in ((2,2,2),(4,4,4)):
            print(f'\ncase: {cell} {cutoff} {upper_corner}')
            atoms = Atoms()
            atoms.lattice_positions(upper_corner=upper_corner,cell=cell)
            if cell == 'fcc' and cutoff == 2.0 and upper_corner == (4,4,4):
                print('stop')
            sg = HilbertGrid(atoms, cellwidth=2.0, cutoff=cutoff)
            vl = sg.sort()
            print(vl)
            pairs = et_md3.verletlist.vl2set(vl)
            # vlsimple must be computed after the spatial sortint
            # because the sorting alters the order of the atoms.
            vlsimple = type(vl)(cutoff)
            et_md3.verletlist.vlbuilders.build_simple(vlsimple, atoms.r)
            print(vlsimple)
            expected = et_md3.verletlist.vl2set(vlsimple)
            assert pairs == expected


def test_HilbertGrid_nonideal_boxes():
    """test with non-ideal boxes: not cubic, or the number of cells in at least onee direction is not a power of 2."""
    for impl in ('py','vl_cpp','cpp'):
        for cell in ('primitive','bcc','fcc'):
            for upper_corner in ((2,2,4),(3,3,3)):
                print(f'\ncase: {impl} {cell} {upper_corner}')
                atoms = Atoms(4)
                atoms.lattice_positions(upper_corner=upper_corner,cell=cell)
                # if impl=='cpp' and cell == 'primitive'  and upper_corner == (2,2,4):
                #     print('stop')
                sg = HilbertGrid(atoms, cellwidth=2.0, impl=impl)
                vl = sg.sort()
                if impl == 'py':
                    print(vl)
                else:
                    vl.print()
                pairs = et_md3.verletlist.vl2set(vl)
                # vlsimple must be computed after the spatial sortint
                # because the sorting alters the order of the atoms.
                vlsimple = type(vl)(sg.cutoff)
                et_md3.verletlist.vlbuilders.build_simple(vlsimple, atoms.r)
                print(vlsimple)
                expected = et_md3.verletlist.vl2set(vlsimple)
                assert pairs == expected


def test_build_hl_vl_cpp():
    global build_hl
    atoms = Atoms(4)
    for cell in ('primitive', 'bcc', 'fcc'):
        for upper_corner in [(2.,2.,4.),(3.,3.,3.)]:
            print(f'\ncase: {cell} {upper_corner}')
            print('\ntesting build_hl')
            atoms.lattice_positions(upper_corner=upper_corner, cell=cell)
            sg = HilbertGrid(atoms, cellwidth=2.0, impl='cpp')
            sg.sort(sort_only=True)

            hl_offset0, hl_natoms0 = build_hl['py'](sg.h) # reference python implementation

            ncells = np.max(sg.h) + 1
            hl_offset = np.empty(ncells, dtype=HilbertGrid.I_t)
            hl_natoms = np.empty(ncells, dtype=HilbertGrid.I_t)

            build_hl['cpp'](sg.h, hl_offset, hl_natoms)  # cpp implementation
            print( f'hl_offset0={hl_offset0}' )
            print( f'hl_offset ={hl_offset}' )
            print( f'hl_natoms0={hl_natoms0}' )
            print( f'hl_natoms ={hl_natoms}' )

            assert np.all(hl_natoms == hl_natoms0)
            assert np.all(hl_offset == hl_offset0)

            # test build_vl_cpp
            print('\ntesting build_vl')
            vl0 = sg.build_vl(sg.h, hl_offset0, hl_natoms0) # reference python implementation
            print('vl0 (python)')
            vl0.print()
            pairs0 = et_md3.verletlist.vl2set(vl0)

            VerletList = et_md3.verletlist.implementation('cpp')
            vl = VerletList(sg.cutoff)
            vl.allocate_2d(sg.atoms.n)
            et_md3.verletlist.vlbuilders.hilbertgrid.spatialsorting.build_vl( sg.K, sg.h, hl_offset, hl_natoms
                       , atoms.r, vl
                       )  # cpp implementation
            print('vl (C++)')
            vl.print()
            pairs =et_md3.verletlist.vl2set(vl)

            assert pairs == pairs0

def test_HilbertGrid_555():
    atoms = Atoms()
    atoms.lattice_positions(upper_corner=(5,5,5))
    for impl in ('py', 'vl_cpp', 'cpp'):
        print(impl)
        sg = HilbertGrid(atoms=atoms,impl=impl)
        vl = sg.sort()

# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_HilbertGrid_555

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
