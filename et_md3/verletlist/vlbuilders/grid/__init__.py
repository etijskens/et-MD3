# -*- coding: utf-8 -*-

"""
Module et_md3.verletlist.vlbuilders.grid
========================================

Grid class for binning atoms. To be used by a vlbuilder
"""

import numpy as np


class Grid:
	"""Construct a Grid object over a rectangular domain of lenght wx
	and height wy.

	:param float cell_size: the width of the square cell
	:param Atoms atoms: atoms object.

	The grid is aligned with the coordinate axes. The axes coincide with the cell
	boundaries. An atom i with position (rxi,ryi,rzi) belongs to cell (k,l,m) iff:

		k*cell_size <= rxi < (k+1)*cell_size

	and

		l*cell_size <= ryi < (l+1)*cell_size

	and

		m*cell_size <= rzi < (m+1)*cell_size

	Note, that we have a <= sign on the left inequality and a < on the right inequality.
	This guarantees that an atom can only belong to a single cell.
	Consequently, the cell of atom i is found as:

		k(i) = floor(rxi/cell_size)
		l(i) = floor(ryi/cell_size)
		m(i) = floor(rzi/cell_size)

	Data structure:

	Similar to the Verlet list approach we use a numpy array for the cell lists.
	There is one Verlet list for each atom i, hence we used a 2D numpy array with
	one row per atom, containing first the number of neighours nn and then the nn
	indices af the atoms in the verlet list of atom i. Here, we use a similar
	approach. Since an atom is identified by a single index and a cell by 3
	indices, we now need at 4D numpy array, with a row of atom indices per cell
	(k,l,m):

	* cl(k,l,m,0) contains the number of atoms in cell (k,l)
	* cl(k,l,m,1..n) are the indices of the atoms in cell (k,l)

	To facilitate passing the cell lists to Fortran routines, we will need to
	linearise the cl data structure. The linearised data structure should allow
	easy lookup of the cell list for a cell given its cell indices (k,l,m).
	This was different for the Verlet list. Our linearised data structure for
	the verlet list does not allow to easily obtain the Verlet list of atom i,
	because we do not know where it starts. We only know how long it is. The only
	way to know where it starts, is to sum the lengths of all Verlet lists of
	atoms before i. This presented no problem for computating the interactions
	because we always need all interactions and thus we start at i=0, loop over
	Verlet list of atom 0, next, we move to atom 1, whose Verlet list come right
	after that of atom 0, and so on. The starting point of each Verlet list is
	easily computed on the fly.

	For the cell lists the situation is different. E.g., to build the Verlet list
	of atom i, which is in cell (k,l,m), we need to access the cell lists

		(k  , l  , m  )
		(k+1, l  , m  )
		(k-1, l+1, m  )
		(k  , l+1, m  )
		(k+1, l+1, m  )
		(k-1, l-1, m+1)
		(k  , l-1, m+1)
		(k=1, l-1, m+1)
		(k-1, l  , m+1)
		(k  , l  , m+1)
		(k=1, l  , m+1)
		(k-1, l+1, m+1)
		(k  , l+1, m+1)
		(k+1, l+1, m+1)

	The problem is easily solved by adding an
	extra array with the starting position of each verlet list. So we have:

	*   cl_list : 1D numpy array containing all cell lists, one after the other,
				  using row-major ordering (https://en.wikipedia.org/wiki/Row-_and_column-major_order)
	*   cl_size : 1D numpy array containing the number of atoms in the cells, i.e.
				  the length of each cell list, also using row-major ordering.
	*   cl_offset : 1D numpy array containing the starting position of all the
				  cell lists in the cl_list array (using row-major ordering)

	So, the cell list of cell (k,l,m) starts at cl_list[rmi(k,l,m)], with rmi(k,l,m) the
	row-major linear index of  (k,l,m) and ends at cl_list[rmi(k,l,m) + cl_size[rmi(k,l,m)]-1 ].

	The indices above are local grid indices. If we are dealing with more than one process, every
	process has its own box, its own local grid. It is assumed that these grids fit nicely together:
	no overlap and no voids.
	The world grid indices are then obtained by adding the offset of the lower_corner of the box.
	"""

	def __init__(self, cell_size, atoms):
		self.cell_size = cell_size
		self.atoms = atoms

		# Compute the grid dimensions K from the atoms box
		self.K = np.ceil((atoms.upper_corner - atoms.lower_corner)/cell_size).astype(int)
		self.ncells = np.prod(self.K)


	def clear(self, zero=False, factor=2):
		"""allocate empty 3D data structure.

		:param bool zero: explicitly assign -1 to all cell list entries. Otherwise, only the
			cell counts are zeroed.
		"""
		max_atoms_per_cell = min(factor * self.atoms.n / self.ncells, self.atoms.n)

		self.cl = np.empty((self.K[0], self.K[1], self.K[2], 1 + max_atoms_per_cell), dtype=int)
		self.cl[:, :, :, 0] = 0 # all cell lists have zero atoms

		if zero:
			self.cl[:, :, 1:] = -1

	def __str__(self):
		s = 'cell lists:\n'
		for m in range(self.K[2]):
			for l in range(self.K[1]):
				for k in range(self.K[0]):
					s += f'({k},{l},{m}) {self.cell_list(k,l,m)}\n'
		s += f'max elements: {self.max_elements()}/{self.max_atoms_per_cell}, linearised={self.linearised()}'
		return s

	def add(self, k, l, m, i):
		"""Add atom i to the cell list of cell (k,l,m)."""
		# increment number of atoms in cell list
		self.cl[k,l,m,0] += 1
		# store the atom in the list at position self.cl[k, l, 0]
		# print( f'VerletList.add: {(k,l)}: {i}')
		self.cl[k,l,m, self.cl[k,l,m, 0]] = i

	def build(self, r=None, linearise=True):
		"""Build cell lists in a straightforward approach.

		:param np.array r: atom positions
		"""
		factor = 2
		try:
			self.clear(factor=factor)
			r = self.atoms.r if r is None else r
			klm = (np.floor(r/self.cell_size)).astype(int)
			# loop over atoms
			self.n_atoms = r.shape[0]  # remember n_atoms for linearization
			for i in range(self.n_atoms):
				self.add(klm[i,0], klm[i,1], klm[i,2], i)

		except IndexError:
			factor *= 1.5

		if linearise:
			self.linearise()


	def max_elements(self):
		"""Return the length of the longest cell list"""
		if self.linearised():
			return np.max(self.cl_size)
		else:
			return np.max(self.cl[:, :, :, 0])


	def linearised(self):
		"""Has this object's data structure been linearised?"""
		return self.cl is None


	def linearise(self):
		"""linearise self.cl."""

		n_cells = self.K[0]*self.K[1]*self.K[2]

		# since every atom belongs to one cell, the length of cl_list is n_atoms
		self.cl_list = np.empty((self.n_atoms,), dtype=int)
		self.cl_size   = np.empty(n_cells, dtype=int)
		self.cl_offset = np.empty(n_cells, dtype=int)

		offset = 0
		for k in range(self.K[0]):
			for l in range(self.K[1]):
				for m in range(self.K[2]):
					n_atoms_in_cell = self.cl[k, l, m, 0]
					# copy the entire cell list at once
					# the cell list is self.cl[k,l,1:1+n_atoms_in_cell]
					self.cl_list[offset:offset+n_atoms_in_cell] = self.cl[k,l,m, 1:1+n_atoms_in_cell]
					# store the current offset for the current cell list
					rmi = row_major_index((k,l,m), self.K)
					self.cl_offset[rmi] = offset
					# store the length of the current cell list
					self.cl_size[rmi] = n_atoms_in_cell
					# move the offset
					offset += n_atoms_in_cell

		# after linearizing we can delete self.cl
		self.cl = None  # Garbage collected

	def cell_list(self, k, l, m):
		"""Obtain the cell list of cell (k,l)."""

		# validate k and l
		if k >= self.K[0]:
			raise IndexError(f'k out of bounds ({k}>={self.K[0]})')
		if l >= self.K[1]:
			raise IndexError(f'l out of bounds ({l}>={self.K[1]})')
		if m >= self.K[2]:
			raise IndexError(f'm out of bounds ({m}>={self.K[2]})')

		if self.linearised():
			rmi = row_major_index((k,l,m), self.K)
			offset = self.cl_offset[rmi]
			n_atoms_in_list = self.cl_size[rmi]
			ckl = self.cl_list[offset:offset+n_atoms_in_list]
		else:
			n_atoms_in_list = self.cl[k, l, m, 0]
			ckl = self.cl[k, l, m, 1:1+n_atoms_in_list]

		return ckl


def row_major_index(k, K):
	"""Convert n-dimensional index k = (k0,k1,...,kn) into a linear index l, using row-major storage'

	(see https://en.wikipedia.org/wiki/Row-_and_column-major_order for details)

	:param list k:	indices
	:param list K: k[i] is the dimension corresponding to k[i]
	:return: linear index of element k
	"""
	l = k[0]
	for ik, iK in zip(k[1:],K[1:]):
		l *= iK
		l += ik
	return l
