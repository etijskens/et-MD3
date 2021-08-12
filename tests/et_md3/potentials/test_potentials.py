#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for sub-module et_md3.potentials."""

import sys
sys.path.insert(0,'.')

import et_md3.potentials
LJ = et_md3.potentials.LennardJones

import numpy as np
import pytest

def test_scaling():
    lj0 = LJ()
    lj1 = LJ(epsilon=1)
    epot0 = lj0.potential(1.)
    epot1 = lj1.potential(1.)
    assert epot1 == 4*epot0
    ff0 = lj0.force_factor(1.)
    ff1 = lj1.force_factor(1.)
    assert ff1 == 4*ff0

    sigma = 2.
    lj2 = LJ(sigma=sigma)
    rij = 1.2
    epot0 = lj0.potential(rij**2)
    epot1 = lj2.potential((rij*sigma)**2)
    # print(epot1,epot0)
    assert epot1 == epot0
    ff0 = lj0.force_factor(rij**2)
    ff1 = lj2.force_factor((rij*sigma)**2)
    # print(ff1,ff0)
    assert ff1*sigma == ff0


def test_potential_minimum():
    """potential has a minimum at r = r0()"""
    lj = LJ()
    r0 = lj.r0()
    vr0 = lj.potential(r0**2)
    d = 0.1
    for i in range(5):
        rLeft  = r0 - d
        rRight = r0 + d
        vLeft  = lj.potential(rLeft**2)
        vRight = lj.potential(rRight**2)
        print(f'{d}: {vLeft} < {vr0} < {vRight}')
        assert vLeft  > vr0
        assert vRight > vr0
        d *= 0.1


def test_potential_zero():
    """potential has a zero at r = 1.0"""
    lj = LJ()
    r =  1.0
    vr = lj.potential(r**2)
    assert vr == 0.0


def test_potential_increasing_right_of_r0():
    """potential increases to the right of r0(), and remains negative."""
    lj = LJ()
    ri = rprev = lj.r0()
    vprev = lj.potential(ri**2)
    for i in range(10):
        ri = 2*ri
        vi = lj.potential(ri**2)
        print(f'{i} {ri}: {vi}')
        assert vprev < vi
        assert vi < 0.0
        vprev = vi
        rprev = ri


def test_potential_decreasing_left_of_r0():
    """potential increases to the right of r0(), and remains negative."""
    lj = LJ()
    ri = rprev = lj.r0()
    vprev = lj.potential(ri**2)
    for i in range(10):
        ri = 0.9*ri
        vi = lj.potential(ri**2)
        print(f'{i} {ri}: {vi}')
        assert vi > vprev
        if ri > 1:
            assert vi < 0.0
        else:
            assert vi > 0.0
        vprev = vi
        rprev = ri


def test_potential_cutoff():
    """not actually a test, just to show the magnitude of the interaction at cut-off."""
    lj = LJ()
    for i in range(1,11):
        rc = i*lj.r0()
        vrc = lj.potential(rc**2)
        print(f'{i} {rc}: `{vrc} {"cut-off" if i==3 else ""}')


def random_unit_vector(n=1):
    """"""
    theta = (2.0*np.py)*np.random.random(n)
    xij = np.cos(theta)
    yij = np.sin(theta)


def test_zero_force_r0():
    """verify that the force magnitude is zero at r0()."""
    lj = LJ()
    rij2 = lj.r0()**2
    fij = lj.force_factor(rij2)
    # account for round-off error r0()**6 is not exactly 5, although r0() is defined as 2**1/6
    assert fij == pytest.approx(0.0, 5e-16)


def test_force_is_derivative_of_potential():
    n = 1000
    # Generate n random numbers in ]0,5*r0()]
    # np.random.random generate numbers in [
    lj = LJ()
    rij = (5*lj.r0())*(1.0 - np.random.random(n))
    fij = lj.force_factor(rij**2)*rij
    d = 1e-10
    rij0 = rij - d
    vij0 = lj.potential(rij0**2)
    rij1 = rij + d
    vij1 = lj.potential(rij1**2)
    dvij = (vij1 - vij0)/(2*d)
    for i in range(n):
        print(f'{i} {fij[i]} == {dvij[i]} {np.abs(fij[i]-dvij[i])}')
        assert fij[i] == pytest.approx(dvij[i],1e-4)


def test_force_direction():
    """test that atoms nearer than r0() are repelled and
    that atomsw farther than r0() are attracted
    """
    lj = LJ()
    rij = (0.9 * lj.r0(), 0, 0)
    fij= lj.force(rij)
    assert fij[0] < 0
    rij = (1.1 * lj.r0(), 0, 0)
    fij = lj.force(rij)
    assert fij[0] > 0


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_scaling

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
