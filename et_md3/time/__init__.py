# -*- coding: utf-8 -*-

"""
Module et_md3.time
==================

Submodule for time and time integration
"""

import numpy as np

class VelocityVerlet:
    """Velocity Verlet algorithm.

    1. compute midstep velocities, at  t + dt/2
    2. compute positions at t + dt
    3. compute accelerations (a = F/m) (external step)
    4. compute velocities at t + dt, and update t = t + dt

    :param impl: 'python'*|'cpp'|'f90': implementation used
    The current time is updated in step_4, when all quantities are known at t
    """
    def __init__(self, atoms=None, r=None, v=None, a=None, impl='python'):
        if not atoms is None:
            self.r = atoms.r
            self.v = atoms.v
            self.a = atoms.a

        else:
            self.r = r # position
            self.v = v # velocity
            self.a = a # acceleration

        if self.r is None or self.v is None or self.a is None:
            raise TypeError

        self.t = 0.0
        if not impl in ['python']: #,'cpp','f90']:
            raise ValueError(f"Invalid implementation {impl}.")

        self.impl = impl
        if self.impl == 'python':
            self.v_midstep = None
        else:
            self.v_midstep = np.zeros_like(self.v)


    @property
    def n_atoms(self):
        return self.r[0].shape[0]


    def step_12(self, dt):
        """Step 1 and 2 of the velocity Verlet algorithm.

        #. Compute (midstep) velocities at t+dt/2
        #. Compute the positions at t+dt.

        :param float dt: timestep
        """
        if self.impl=='python':
            # Step 1: compute velocities at midstep (t+dt/2) using the current accelerations:
            self.v_midstep = self.v + (0.5*dt)*self.a

            # Step 2: compute positions at next step (t+dt) using the midstep velocities:
            self.r += self.v_midstep*dt

        elif self.impl=='cpp':
            raise NotImplementedError
            # et_ppmd.corecpp.velocity_verlet_12( dt
            #                                   , self.rx, self.ry
            #                                   , self.vx, self.vy
            #                                   , self.ax, self.ay
            #                                   , self.vx_midstep, self.vy_midstep
            #                                   )
        elif self.impl=='f90':
            raise NotImplementedError
            # et_ppmd.coref90.velocity_verlet_12( dt
            #                                   , self.rx, self.ry
            #                                   , self.vx, self.vy
            #                                   , self.ax, self.ay
            #                                   , self.vx_midstep, self.vy_midstep
            #                                   )


    def step_4(self, dt):
        """Step 4 of the velocity Verlet algorithm.

        Compute the velocities at t+dt.

        :param float dt: timestep
        """
        if self.impl=='python':
            # Step 4: compute velocities at next step (t+dt)
            self.v = self.v_midstep + self.a * (0.5*dt)

        elif self.impl=='cpp':
            raise NotImplementedError
            # et_ppmd.corecpp.velocity_verlet_4( dt
            #                                 , self.rx, self.ry
            #                                 , self.vx, self.vy
            #                                 , self.ax, self.ay
            #                                 # , self.vx_midstep, self.vy_midstep
            #                                 # )

        elif self.impl=='f90':
            raise NotImplementedError
            # et_ppmd.coref90.velocity_verlet_4( dt
            #                                 , self.rx, self.ry
            #                                 , self.vx, self.vy
            #                                 , self.ax, self.ay
            #                                 # , self.vx_midstep, self.vy_midstep
            #                                 )

        # Update the current time:
        self.t += dt
