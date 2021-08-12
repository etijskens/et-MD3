#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et-MD` package."""

import sys
sys.path.insert(0,'.')

import pytest
import numpy as np

from et_md3.atoms import Atoms
from et_md3.time import VelocityVerlet
import et_md3.potentials
# import matplotlib.pyplot as plt


def test_VelocityVerlet_no_motion():
    atoms = Atoms(2,zero=True)
    vv = VelocityVerlet(atoms=atoms)
    dt = 1.0
    for t in range(10):
        vv.step_12(dt)
        print(vv.r)
        assert np.all(vv.r == 0.0)
        assert np.all(vv.v == 0.0)
        assert np.all(vv.a == 0.0)
        # Step 3: no forces
        vv.step_4(dt)
        assert np.all(vv.r == 0.0)
        assert np.all(vv.v == 0.0)
        assert np.all(vv.a == 0.0)

def test_VelocityVerlet_constant_velocity():
    atoms = Atoms(3,zero=True)
    atoms.v[0,:] = np.array([1,0,0], dtype=float)
    atoms.v[1,:] = np.array([0,1,0], dtype=float)
    atoms.v[2,:] = np.array([0,0,1], dtype=float)

    vv = VelocityVerlet(atoms=atoms)

    dt = 1.0
    for it in range(10):
        vv.step_12(dt)
        assert vv.r[0][0] == 1.0 * dt * (it + 1)
        assert vv.r[0][1] == 0.0
        assert vv.r[0][2] == 0.0

        assert vv.r[1][0] == 0.0
        assert vv.r[1][1] == 1.0 * dt * (it + 1)
        assert vv.r[1][0] == 0.0

        assert vv.r[2][0] == 0.0
        assert vv.r[2][1] == 0.0
        assert vv.r[2][2] == 1.0 * dt * (it + 1)

        assert vv.v[0][0] == 1.0
        assert vv.v[0][1] == 0.0
        assert vv.v[0][2] == 0.0

        assert vv.v[1][0] == 0.0
        assert vv.v[1][1] == 1.0
        assert vv.v[1][2] == 0.0

        assert np.all(vv.a[0] == 0.0)
        assert np.all(vv.a[1] == 0.0)
        assert np.all(vv.a[2] == 0.0)

        # Step 3: no forces

        vv.step_4(dt)
        assert vv.r[0][0] == 1.0 * dt * (it + 1)
        assert vv.r[0][1] == 0.0
        assert vv.r[0][2] == 0.0

        assert vv.r[1][0] == 0.0
        assert vv.r[1][1] == 1.0 * dt * (it + 1)
        assert vv.r[1][2] == 0.0

        assert vv.r[2][0] == 0.0
        assert vv.r[2][1] == 0.0
        assert vv.r[2][2] == 1.0 * dt * (it + 1)

        assert vv.v[0][0] == 1.0
        assert vv.v[0][1] == 0.0
        assert vv.v[0][2] == 0.0

        assert vv.v[1][0] == 0.0
        assert vv.v[1][1] == 1.0
        assert vv.v[1][2] == 0.0

        assert vv.v[2][0] == 0.0
        assert vv.v[2][1] == 0.0
        assert vv.v[2][2] == 1.0

        assert np.all(vv.a[0] == 0.0)
        assert np.all(vv.a[1] == 0.0)
        assert np.all(vv.a[2] == 0.0)


def test_VelocityVerlet_constant_acceleration():
    """analytical solution:
    a = cst
    v = a*t
    r = 0.5*a*t**2
    """
    atoms = Atoms(3,zero=True)
    a = 1.0
    atoms.a[0,:] = np.array([a,0,0], dtype=float)
    atoms.a[1,:] = np.array([0,a,0], dtype=float)
    atoms.a[2,:] = np.array([0,0,a], dtype=float)

    vv = VelocityVerlet(atoms=atoms)
    dt = 1e-4
    nsteps = 10000
    for it in range(nsteps):
        vv.step_12(dt)
        t = (it+1)*dt
        print(f't={t}')
        r = 0.5 * a * t**2
        print(f'r={r}')
        assert vv.r[0][0] == pytest.approx(r, 1e-8)
        assert vv.r[0][1] == 0.0
        assert vv.r[0][2] == 0.0

        assert vv.r[1][0] == 0.0
        assert vv.r[1][1] == pytest.approx(r, 1e-8)
        assert vv.r[1][2] == 0.0

        assert vv.r[2][0] == 0.0
        assert vv.r[2][1] == 0.0
        assert vv.r[2][2] == pytest.approx(r, 1e-8)

        # Step 3: no forces

        vv.step_4(dt)
        v = a*t
        print(f'v={v}\n')

        assert vv.v[0][0] == pytest.approx(v, 1e-8)
        assert vv.v[0][1] == 0.0
        assert vv.v[0][2] == 0.0

        assert vv.v[1][0] == 0.0
        assert vv.v[1][1] == pytest.approx(v, 1e-8)
        assert vv.v[1][2] == 0.0

        assert vv.v[2][0] == 0.0
        assert vv.v[2][1] == 0.0
        assert vv.v[2][2] == pytest.approx(v, 1e-8)

    # test the current time:
    assert vv.t == pytest.approx(dt*nsteps,1e-8)

# def test_VelocityVerlet_constant_acceleration_2():
#     """analytical solution:
#     a = cst
#     v = a*t
#     r = 0.5*a*t**2
#     """
#     a = 1.0
#     for impl in ['f90','cpp']:
#         print(f'testing {impl} implementation')
#         ry = np.zeros((2,), dtype=float)
#         vx = np.zeros((2,), dtype=float)
#         rx = np.zeros((2,), dtype=float)
#         vy = np.zeros((2,), dtype=float)
#         ax = np.array([a,0], dtype=float)
#         ay = np.array([0,a], dtype=float)
#         vv = VelocityVerlet(rx, ry, vx, vy, ax, ay,impl=impl)
#         dt = 1e-4
#         nsteps = 10000
#         for it in range(nsteps):
#             vv.step_12(dt)
#             t = (it+1)*dt
#             print(f't={t}')
#             r = 0.5 * a * t**2
#             print(f'r={r}')
#             assert vv.rx[0] == pytest.approx(r, 1e-8)
#             assert vv.rx[1] == 0.0
#             assert vv.ry[0] == 0.0
#             assert vv.ry[1] == pytest.approx(r, 1e-8)
#
#             # Step 3: no forces
#
#             vv.step_4(dt)
#             v = a*t
#             print(f'v={v}\n')
#             assert vv.vx[0] == pytest.approx(a * t, 1e-8)
#             assert vv.vx[1] == 0.0
#             assert vv.vy[0] == 0.0
#             assert vv.vy[1] == pytest.approx(a * t, 1e-8)
#
#         # test the current time:
#         assert vv.t == pytest.approx(dt*nsteps,1e-8)

# def test_2particles():
#     dt = 1e-2
#     nsteps = 10000
#     impl = 'f90' #,'f90','cpp']:
#     rx = np.zeros((2,), dtype=float)
#     ry = np.zeros((2,), dtype=float)
#     vx = np.zeros((2,), dtype=float)
#     vy = np.zeros((2,), dtype=float)
#     ax = np.zeros((2,), dtype=float)
#     ay = np.zeros((2,), dtype=float)
#     vv = VelocityVerlet(rx, ry, vx, vy, ax, ay, impl=impl)
#     delta = .1
#     rx[0] = -(et_ppmd.forces.R0/2) - delta
#     rx[1] =  (et_ppmd.forces.R0/2) + delta
#     xt1 = np.zeros((nsteps,),dtype=float)
#     for it in range(nsteps):
#         vv.step_12(dt)
#
#         x01 = rx[1] - rx[0]
#         y01 = ry[1] - ry[0]
#         f = et_ppmd.forces.force(x01,y01)
#         ax[0] =  f[0]
#         ax[1] = -f[0]
#         ay[0] =  f[1] # always zero
#         ay[1] = -f[1] # always zero
#
#         vv.step_4(dt)
#
#         xt1[it] = rx[1] - et_ppmd.forces.R0/2
#     plt.plot(xt1)
#     plt.show()


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_VelocityVerlet_constant_acceleration

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
