# -*- coding: utf-8 -*-

"""
Module et_md3.potentials
========================

A submodule for interaction potentials.

"""
import numpy as np


class Potential:
	"""Base class for interaction potential classes. It provides the machinery to
	compute the interaction energy and interaction forces for a Verlet list.

	Derived classes must expose:

	* force_factor(rij2): the force fij = dV(|rij|)/drij, and is equal to rij * force_factor(rij2)
	* potential(rij2): V(rij2)

	Here,

	* rij is the vector rj - ri
	* |rij| is the magnitude of rij
	* rij2 is the dot product rij.rij = |rij|**2
	* V(rij) is the interaction potential as a function of rij
	* V(rij2) is the interaction potential as a function of rij2
	"""


	def compute_interaction_forces(self, vl, r, a):
		"""Compute interaction forces from the atom pairs in the Verlet list.

		The interaction forces are added to a. At the beginning of each time step,
		a must be zeroed.

		:param et_md3.verletlist.VL vl; Verlet list
		:param np.ndarray r: atom position coordinates
		:param np.ndarray a: atom acceleration coordinates
		"""
		for i in range(vl.natoms):
			o = vl.vl_offset[i]
			n = vl.vl_size[i]
			ri = r[i, :]
			ai = a[i, :]
			for k in range(o, o + n):
				j = vl.vl_list[k]
				rij = r[j, :] - ri
				rij2 = np.dot(rij, rij)
				rij *= self.force_factor(rij2)
				ai += rij
				a[j, :] -= rij


	def compute_interaction_energy(self, vl, r):
		"""Compute the interaction energy from the atom npairs Verlet List.

		:param et_md3.verletlist.VL vl; Verlet list
		:param np.ndarray r: atom coordinates
		:return: interaction energy, epot.
		"""
		epot = 0.0
		for i in range(vl.natoms):
			o = vl.vl_offset[i]
			n = vl.vl_size[i]
			for k in range(o, o + n):
				j = vl.vl_list[k]
				rij = r[j, :] - r[i, :]
				rij2 = np.dot(rij, rij)
				epot += self.potential(rij2)

		return epot


class LennardJones(Potential):
    """Lennard-Jones potentiol

    See https://en.wikipedia.org/wiki/Lennard-Jones_potential.

    V_lj(r) = 4*epsilon*[ (sigma/r)**12 - (sigma/r)**6 ]

    :param float epsilon: epsilon
    :param float sigma: sigma
    """
    def __init__(self,epsilon=0.25, sigma=1):
        self.four_epsilon = 4.0*epsilon
        self.inv_sigma = (1.0/sigma)


    def r0(self):
        """Equilibrium distance."""
        return pow(2.,1/6) / self.inv_sigma


    def potential(self, rij2):
        """Compute the Lennard-Jones potential

        :param float|np.array rij2: squared distance between atoms.
        :returns: a float.
        """
        rij_sigma_2 = rij2 * self.inv_sigma * self.inv_sigma
        rm6 = 1./(rij_sigma_2*rij_sigma_2*rij_sigma_2)
        vlj = self.four_epsilon*(rm6 - 1.0)*rm6
        return vlj


    def force_factor(self, rij2):
        """Lennard-Jones force magnitude exerted by atom j on atom i.

        :param float|np.array rij2: squared interatomic distance from atom i to atom j
        :return: fij
        """
        rij_sigma_2 = rij2 * self.inv_sigma * self.inv_sigma
        rm2 = 1.0 / rij_sigma_2
        rm6 = (rm2 * rm2 * rm2)
        f = self.four_epsilon * self.inv_sigma * 6.0 * (1.0 - 2.0 * rm6) * rm6 * rm2
        return f


    def force(self, rij):
        """Lennard-Jones force exerted by atom j on atom i.

        :param 3-tuple of float|np.array rij: vector from atom i to atom j
        :return: Fij, list of float|np.array
        """
        rij2 = rij[0] ** 2 + rij[1] ** 2 + rij[2] ** 2
        f = self.force_factor(rij2)
        return ( f * rij[0]
               , f * rij[1]
               , f * rij[2]
               )
