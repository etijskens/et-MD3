#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for C++ module et_md3.atoms.cpp.
"""

import sys

import numpy as np

import et_md3.atoms
import et_md3.atoms.atoms_cpp

def test_scale_forces():
    atms = et_md3.atoms.Atoms(5)
    for impl in ('cpp', 'py'):
        for i in range(5):
            atms.a[i,:] = 1+i
            atms.m[i]   = 1+i
        print(atms.a)
        atms.scale_forces(impl=impl)
        print(atms.a)
        assert np.all(atms.a == 1.)

#===============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
#===============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_scale_forces

    print(f"__main__ running {the_test_you_want_to_debug} ...")
    the_test_you_want_to_debug()
    print('-*# finished #*-')
#===============================================================================
