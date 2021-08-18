# -*- coding: utf-8 -*-

"""
Module et_md3.verletlist.vlbuilders.hilbertgrid
===============================================

This module provide a spatially sorted grid, using Hilbert indices. This means that all
atom property arrays are sorted according to the hilbert index of the grid cell in which
the atoms reside. All atoms in the same cell thus appear contiguously in atom property
arrays. This allows for a simple grid implementation. All that is needed is a map from
Hilbert indices to the offset of the grid cell in the atom property arrays, and a map
from Hilbert indices to the number of atoms in the cell. A grid list is no longer needed.

"""
import numpy as np

import et_md3.verletlist
import et_md3.verletlist.vlbuilders.hilbertgrid.spatialsorting
from math import sqrt

# spatialsorting = et_md3.verletlist.vlbuilders.hilbertgrid.spatialsorting

def hilbert_cube(k):
    """Find the smallest power of 2 that is greater or equal than k. This is the
    side s of a complete hilbert cube. The cube contains all hilbert indices from
    0 to s**3 - 1.

    :param int k: a positive number, usually the maximum of a HilbertGrid's K member.
    :return: int
    """
    if k <= 2:
        return k

    k2 = 4
    while True:
        if k2 >= k:
            return k2
        k2 *= 2


class HilbertGrid:
    """This class imposes a grid over an Atoms' box and uses it for spatial sorting,
    using Hilbert indices, as well as for building the Verlet list.

    The class has an additional C++ implementation for building the Verlet list. The Verlet
    list may be the Python version or its C++ equivalent

    :param Atoms atoms:
    :param float cellwidth: width of the grid cells,, If None (default), the cellwidth is set
        equal to the cutoff distance. If that is None too, both are set equal to 1.0.
    :param float cutoff: cutoff distance for the Verlet list. If None (default), the cutoff
        distance is set equal to the grid cell width. If that is None too, both are set equal
        to 1.0.
    """
    # These two types are used internally
    H_t = np.longlong # type for hilbert indices
    I_t = np.uint32   # type for array indices
    ijk_t = np.intc   # type for cell indices

    def __init__( self, atoms, cellwidth=None, cutoff=None
                , impl='py'
                ):
        impls = ( 'py'       # all python (except for the hilbert indices)
                , 'vl_cpp'   # using C++ verlet list implementation
                , 'cpp'      # C++ implementation for building the Hilbert list and the Verlet list
                )
        if not impl in impls:
            raise ValueError(f'impl=`{impl}` is not in `{impls}`')
        self.impl = impl
        self.atoms = atoms

        if cellwidth is None:
            if cutoff is None:
                # both are None
                cellwidth, cutoff = 1.0, 1.0
            else:
                # only cutoff is specified
                cellwidth = cutoff
        else:
            if cutoff is None:
                # only cellwidth is specified
                cutoff = cellwidth
            else:
                # both are specified,
                if cutoff > cellwidth:
                    raise ValueError(f'The cutoff distance ({cutoff} is larger than the grid cell width ({cellwidth}.\n'
                                     f'This may lead to missed contacts.')
        self.cellwidth = cellwidth
        self.cutoff = float(cutoff)

        # Compute the grid dimensions K from the atoms box
        self.K = np.ceil((atoms.upper_corner - atoms.lower_corner) / cellwidth).astype(HilbertGrid.I_t)


    def sort(self, sort_only=False):
        """Perform a spatial sort, compute Hilbert list and Verlet Lists.

        :param bool sort_only: if True does not compute Hilbert list and Verlet lists.
            (used for testing).
        :returns: Verlet list object, type depends on implementation choses, or None if
            sort_only is False.
        """

        # compute the hilbert index of every atom ------------------------------
        h = np.empty(self.atoms.n, dtype=HilbertGrid.H_t)
        spatialsorting.rw2h_dp( self.atoms.r, self.cellwidth, h )

        # perform the spatial sort ---------------------------------------------
        I = np.empty(self.atoms.n, dtype=HilbertGrid.I_t)

        spatialsorting.sort(h,I)
        # sort the atom arrays:
        ar_sorted = np.empty_like(self.atoms.arrays[0])
        for ar in self.atoms.arrays:
            d = len(ar.shape)
            if d == 1:
                if ar.dtype == np.double:
                    reorder = spatialsorting.reorder_dp
                elif ar.dtype == np.single:
                    reorder = spatialsorting.reorder_sp
                elif ar.dtype == HilbertGrid.H_t:
                    reorder = spatialsorting.reorder_uint32
                elif ar.dtype == HilbertGrid.I_t:
                    reorder = spatialsorting.reorder_int32
                elif ar.dtype == int:
                    reorder = spatialsorting.reorder_longlongint
                else:
                    raise NotImplementedError(f'spatialsorting.reorder() not implemented for {d}-dimension array of type {ar.dtype}.')
            elif d == 2:
                if ar.dtype == np.double:
                    reorder = spatialsorting.reorder_2d_dp
                else:
                    raise NotImplementedError(f'spatialsorting.reorder() not implemented for {d}-dimension array of type {ar.dtype}.')
            # reuse the sorted array if we can
            if ar.dtype != ar_sorted.dtype or ar.shape != ar_sorted.shape:
                ar_sorted = np.empty_like(ar)

            reorder(I, ar, ar_sorted)
            ar[:] = ar_sorted

        if sort_only: # for testing the c++ implementation
            self.h = h
            return

        # build the hilbert list -----------------------------------------------
        # . a 1d numpy array with the offset of each cell (along the hilbert
        #   curve): self.hl_offset
        # . a 1d numpy array with the number of atoms in each cell (along the
        #   hilbert curve) : self.hl_natoms

        if self.impl == 'cpp':
            ncells = np.max(h) + 1

            hl_offset = np.empty(ncells, dtype=HilbertGrid.I_t)
            hl_natoms = np.empty(ncells, dtype=HilbertGrid.I_t)
            spatialsorting.build_hl(h, hl_offset, hl_natoms)
        else:
            hl_offset, hl_natoms = build_hl(h)

        # build the verlet list ------------------------------------------------
        if self.impl == 'cpp':
            VerletList = et_md3.verletlist.implementation('cpp')
            vl = VerletList(self.cutoff)
            vl.allocate_2d(self.atoms.n)
            spatialsorting.build_vl(self.K, h, hl_offset, hl_natoms, self.atoms.r, vl )
        else:
            vl = self.build_vl(h, hl_offset, hl_natoms)

        return vl


    def build_vl( self, h, hl_offset, hl_natoms):
        """Build the verlet list from the Hilbert list

        :param h: numpy array with Hilbert index of each atom (ordered).
        :param hl_offset: numpy array with the starting point of each grid cell in h.
        :param hl_natoms: numpy array with the number of atoms in the grid cells.
        :param str impl_vl; implementation used for the Verlet_list

        central cell
            (k  , l  , m  )
        neigbour cells to visit:
            (k+1, l  , m  )

            (k-1, l+1, m  )
            (k  , l+1, m  )
            (k+1, l+1, m  )

            (k-1, l-1, m+1)
            (k  , l-1, m+1)
            (k=1, l-1, m+1)
            (k-1, l  , m+1)
            (k  , l  , m+1)
            (k+1, l  , m+1)
            (k-1, l+1, m+1)
            (k  , l+1, m+1)
            (k+1, l+1, m+1)

        """
        ncells = hl_offset.shape[0]
        ijk_central = np.empty(3, dtype=HilbertGrid.ijk_t)
        ijk_delta = np.array(
            [ [ 1, 0, 0] # x-direction
            , [-1, 1, 0] # y-direction
            , [ 0, 1, 0]
            , [ 1, 1, 0]
            , [-1,-1, 1] # z-direction
            , [ 0,-1, 1]
            , [ 1,-1, 1]
            , [-1, 0, 1]
            , [ 0, 0, 1]
            , [ 1, 0, 1]
            , [-1, 1, 1]
            , [ 0, 1, 1]
            , [ 1, 1, 1]
            ]
            , dtype=HilbertGrid.ijk_t
        )
        ijk_nb = np.empty_like(ijk_delta)

        cutoff2 = self.cutoff**2

        if self.impl == 'py':
            VerletList = et_md3.verletlist.implementation('py')
            vl = VerletList(self.cutoff)
            vl.allocate_2d(self.atoms.n)
        elif self.impl in ('vl_cpp','cpp'):
            VerletList = et_md3.verletlist.implementation('cpp')
            vl = VerletList(self.cutoff)
            vl.allocate_2d(self.atoms.n)
        else:
            raise NotImplementedError(f'impl_vl={self.impl}')

        # problem: only if the box is cubic with size a power of 2 cells in each direction
        # the number of cells bounds the hilbert indices.
        # in more general cases some hilbert indice may exceed the number of cells.

        # Loop over all cells:
        h_prev = -1
        for h_central in h:
            if h_central == h_prev:
                continue # each cell must be visited only once
            else:
                h_prev = h_central

            # compute the central cell's indices
            spatialsorting.h2ijk(h_central, ijk_central)
            #print(f'h:{h_central} ijk_central{ijk_central}')

            # compute the indices of the neighbouring cells
            ijk_nb = ijk_central + ijk_delta
            # note that the ijk_delta stencil may refer to nonexisting cells (outside the box)

            # ijk_central-ijk_central pairs
            offset_central = hl_offset[h_central]
            natoms_central = hl_natoms[h_central]
            for i in range(offset_central,offset_central+natoms_central-1):
                ri = self.atoms.r[i]
                for j in range(i+1,offset_central+natoms_central):
                    rj = self.atoms.r[j]
                    rij = rj - ri
                    rij2 = np.dot(rij,rij)
                    if rij2 <= cutoff2:
                        # print(f'c-c {(i, j)}')
                        vl.add(i,j)

            # ijk_central-ijk_nb pairs
            for nb in range(13):
                if  -1 < ijk_nb[nb,0] < self.K[0] \
                and -1 < ijk_nb[nb,1] < self.K[1] \
                and -1 < ijk_nb[nb,2] < self.K[2]:
                    print(f'nb={nb} {ijk_nb[nb,:]}, {self.K}')
                    # Otherwise the cell is outside the box
                    h_nb = spatialsorting.ijk2h_1(ijk_nb[nb,:])
                    #print(f'h_nb:{h_nb} ijk_nb{ijk_nb[nb,:]}')
                    offset_nb = hl_offset[h_nb]
                    natoms_nb = hl_natoms[h_nb]
                    for i in range(offset_central, offset_central + natoms_central):
                        ri = self.atoms.r[i]
                        for j in range(offset_nb, offset_nb + natoms_nb ):
                            rj = self.atoms.r[j]
                            rij = rj - ri
                            rij2 = np.dot(rij,rij)
                            if rij2 <= cutoff2:
                                # print(f'c-n {(i, j)}')
                                vl.add(i, j)

        vl.linearise(False)
        return vl

def build_hl(h):
    """Build hilbert list"""
    natoms = h.shape[0]
    ncells = np.max(h)+1
    hl_offset = np.empty(ncells,dtype=HilbertGrid.I_t)
    hl_natoms = np.empty(ncells,dtype=HilbertGrid.I_t)
    ia0, ia = 0, 0
    hl_offset[ia] = 0
    for hi in range(ncells):
        while ia<natoms and h[ia] == hi:
            ia += 1
        else:
            hl_offset[hi] = ia0
            hl_natoms[hi] = ia - ia0
            ia0 = ia
            # print(hi, hl_offset[hi], hl_natoms[hi])
        if ia >= natoms:
            break
    return hl_offset, hl_natoms
