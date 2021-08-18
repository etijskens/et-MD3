/*
 *  C++ source file for module et_md3.verletlist/vlbuilders/hilbertgrid/spatialsorting
 *  Created on: 15 Sep 2016
 *      Author: etijskens
 *
 * Python wrapper for hilbert indices
 * code is in ./src as recuperated from hpc-tnt-1.2 (2016)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// include source code
#include "./src/hilbert_c.cpp"
#include "./src/hilbert.cpp"
// some abbreviations:
typedef hilbert::HilbertIndex_t H_t; // type for hilbert indices
typedef hilbert::I_t            I_t; // type for indexing arrays

// common code for dealing with numpy arrays
#include "../../../../pybind11/ArrayInfo.hpp"

#include <limits>
// for debugging mainly
#include <iostream>
#include <cassert>
/*
// original (hpc-tnt-1.2) code for building verlet lists.
// this can probably be more efficient than using numpy arrays
// for the data structure
namespace helper
{//-------------------------------------------------------------------
    template <class T>
    inline double sq(T const t ) {
        return t*t;
    }

    int E[13][2][3] =   { {{0,0,0},{1,0,0}}, //0
                          {{0,0,0},{0,1,0}},
                          {{0,0,0},{0,0,1}},
                          {{0,0,0},{1,1,0}},
                          {{0,0,0},{1,0,1}},
                          {{0,0,0},{0,1,1}},
                          {{0,0,0},{1,1,1}},
                          {{1,0,0},{0,1,0}}, //7
                          {{1,0,0},{0,0,1}},
                          {{1,0,0},{0,1,1}},
                          {{0,1,0},{0,0,1}}, //10
                          {{0,1,0},{1,0,1}},
                          {{0,0,1},{1,1,0}}  //12
                        };

    class VerletListBuilder
    {
    public:
     // Ctor
        VerletListBuilder
          ( ArrayInfo<int   ,2>& verlet_list
          , ArrayInfo<int   ,2>& hl_
          , ArrayInfo<double,1>& rx
          , ArrayInfo<double,1>& ry
          , ArrayInfo<double,1>& rz
          , ArrayInfo<int   ,1> I
          , double                rcutoff2
          )
          : n_atoms_(verlet_list.shape()[1])
          , reserve_(verlet_list.shape()[0])
          , rx_(rx.origin())
          , ry_(ry.origin())
          , rz_(rz.origin())
          , I_ (( I.shape()[0]==0 ? nullptr : I.origin() ))
          , vl_(verlet_list)
          , hl_(hl_)
          , rcutoff2_(rcutoff2)
        {
            this->construct();
        }

        void add(int const i, int const j )
        {
            if(j<i) {
                this->vl_[0][i] += 1;
                int const n = this->vl_[0][i];
                this->vl_[n][i] = j;
            } else {
                this->vl_[0][j] += 1;
                int const n = this->vl_[0][j];
                this->vl_[n][j] = i;
            }
        }

        inline int ia2i( int ia ) const {
            return ( this->I_==0 ? ia           // this->I_ is nullptr
                                 : this->I_[ia] // this->I_ is nonzero pointer
                   );
        }

        void add_cell( H_t h0 )
        {// update verlet lists for h0-h0 interactions (intra-cell)
            int const first_atom_in_h0 = this->hl_[h0][0];
            int const n_atoms_in_h0    = this->hl_[h0][1];
            for( int ia=first_atom_in_h0+1; ia<first_atom_in_h0+n_atoms_in_h0; ++ia )
            {
                int const i = ia2i(ia);
                double xi = this->rx_[i];
                double yi = this->ry_[i];
                double zi = this->rz_[i];
                for( int ja=first_atom_in_h0; ja<ia; ++ja )
                {
                    int const j = ia2i(ja);
                    double const r2 = sq(xi-this->rx_[j]) + sq(yi-this->ry_[j]) + sq(zi-this->rz_[j]);
                    if( r2<=this->rcutoff2_ )
                        this->add(i,j);
                }
            }
        }

        void add_cell_cell( H_t h0, H_t h1 )
        {// update verlet lists for h0-h1 interactions (h0!=h1)
            if( h0==h1 )
                std::cout<<"oops "<<h0<<"-"<<h1<<std::endl;
            assert(h0!=h1);
            int first_atom_in_h0 = this->hl_[h0][0];
            int n_atoms_in_h0    = this->hl_[h0][1];
            int first_atom_in_h1 = this->hl_[h1][0];
            int n_atoms_in_h1    = this->hl_[h1][1];
            for( int ia=first_atom_in_h0; ia<first_atom_in_h0+n_atoms_in_h0; ++ia )
            {
                int const i = ia2i(ia);
                double xi = this->rx_[i];
                double yi = this->ry_[i];
                double zi = this->rz_[i];
                for( int ja=first_atom_in_h1; ja<first_atom_in_h1+n_atoms_in_h1; ++ja )
                {
                    int const j = ia2i(ja);
                    double const r2 = sq(xi-this->rx_[j]) + sq(yi-this->ry_[j]) + sq(zi-this->rz_[j]);
                    if( r2<=this->rcutoff2_ ) {
                        this->add(i,j);
                    }
                }
            }
        }

        H_t h_neighbour( int const ijk[3], int i0E, int i1E) {
            int ijkE[3];
            for( int c=0; c<3; ++c ) {
                ijkE[c] = ijk[c] + E[i0E][i1E][c];
            }
            return hilbert::ijk2h(ijkE);
        }

        void construct()
        {// loop over all cells
            int ijk00[3] = {0,0,0};
            int const nh = hl_.shape()[0];
            for( int h00=0; h00<nh; ++h00 )
            {
             // intra-cell
                this->add_cell(h00);

             // loop over neighbouring cells
                hilbert::h2ijk(h00,ijk00);
                int nb0[5] = {0,7,10,12,13};
                for( int inb0=0; inb0<4; ++inb0 )
                {
                    H_t         h0E = this->h_neighbour(ijk00,nb0[inb0],0);
                    if( -1<h0E && h0E<nh ) {
                        for( int inb1=nb0[inb0]; inb1<nb0[inb0+1]; ++inb1 ) {
                            H_t h1E = this->h_neighbour(ijk00,inb1,1);
                            if( -1<h1E && h1E<nh ) {
                                this->add_cell_cell(h0E,h1E);
                            }
                        }
                    }
                }
            }
        }
    private:
        int n_atoms_;
        int reserve_;
        double *rx_, *ry_, *rz_;
        int const * const I_;
        py::array_t<int,2>& vl_;
        py::array_t<int,2>& hl_;
        double rcutoff2_;
    };
 //-------------------------------------------------------------------
}// namespace helper

void
build_verlet_list
  ( py::array_t<int   > verlet_list  // inout
  , py::array_t<int   > hilbert_list // in
  , py::array_t<double> rx           // in
  , py::array_t<double> ry           // in
  , py::array_t<double> rz           // in
  , py::array_t<int   > I            // in
  , double              rcutoff2     // in
  )
{
    ArrayInfo<int   ,2> a_verlet_list(verlet_list);
    ArrayInfo<int   ,2> a_hilbert_list(hilbert_list);
    ArrayInfo<double,1> a_rx(rx);
    ArrayInfo<double,1> a_ry(rx);
    ArrayInfo<double,1> a_rz(rz);

    helper::VerletListBuilder vlb( verlet_list, hilbert_list, rx, ry, rz, I, rcutoff2 );
}
*/

// convert 3D positions (numpy arrays) to hilbert indices
void xyzw2h_float64
  ( py::array_t<double> x // in
  , py::array_t<double> y // in
  , py::array_t<double> z // in
  ,             double  w // in
  , py::array_t<H_t   > h // out
  )
{
    ArrayInfo<double,1> ax(x), ay(y), az(z);
    ArrayInfo<H_t   ,1> ah(h);
    ax.assert_identical_shape(ay);
    ax.assert_identical_shape(az);
    ax.assert_identical_shape(ah);
 /*
    double const *ptrx = ax.cdata();
    double const *ptry = ay.cdata();
    double const *ptrz = az.cdata();
    H_t          *ptrh = ah.data();

    I_t n = ax.shape(0);

    for( I_t i=0; i<n; i++ ) {
        ptrh[i] = hilbert::xyzw2h(ptrx[i],ptry[i],ptrz[i],w);
    }
  */
    I_t const n = ax.shape(0);
    for( I_t i=0; i<n; i++ ) {
        ah[i] = hilbert::xyzw2h(ax[i],ay[i],az[i],w);
    }
}

void rw2h_float64
  ( py::array_t<double> r // in
  ,             double  w // in
  , py::array_t<H_t   > h // out
  )
{
    ArrayInfo<double,2> ar(r);
    ArrayInfo<H_t   ,1> ah(h);

    I_t const n = ar.shape(0);
    I_t const m = ar.shape(1);
    for( I_t i=0; i<n; i++ ) {
        ah[i] = hilbert::rw2h(&ar[m*i], w);
//        std::cout<<"rw2h_float64 "<<i<<" ["<<p[0]<<" "<<p[1]<<" "<<p[2]<<"] ->"<<ah[i]<<std::endl;
    }
}

void xyzw2h_float32
  ( py::array_t<float> x
  , py::array_t<float> y
  , py::array_t<float> z
  ,             float  w
  , py::array_t<H_t>   h
  )
{
    ArrayInfo<float,1> ax(x), ay(y), az(z);
    ArrayInfo<H_t  ,1> ah(h);
    ax.assert_identical_shape(ay);
    ax.assert_identical_shape(az);
    ax.assert_identical_shape(ah);

    I_t const n = ax.shape(0);
    for( I_t i=0; i<n; i++ ) {
        ah[i] = hilbert::xyzw2h(ax[i],ay[i],az[i],w);
    }
}

void rw2h_float32
  ( py::array_t<float> r // in
  ,             float  w // in
  , py::array_t<H_t  > h // out
  )
{
    ArrayInfo<float,2> ar(r);
    ArrayInfo<H_t  ,1> ah(h);

    I_t const n = ar.shape(0);
    I_t const m = ar.shape(1);
    for( I_t i=0; i<n; i++ ) {
        ah[i] = hilbert::rw2h(&ar[m*i], w);
//        std::cout<<"rw2h_float32 "<<i<<" ["<<p[0]<<" "<<p[1]<<" "<<p[2]<<"] ->"<<ah[i]<<std::endl;
    }
}

void xyzw2ijkh_float64
  ( py::array_t<double> x // in
  , py::array_t<double> y // in
  , py::array_t<double> z // in
  ,             double  w // in
  , py::array_t<int>    i // out
  , py::array_t<int>    j // out
  , py::array_t<int>    k // out
  , py::array_t<H_t>    h // out
  )
{
    ArrayInfo<double,1> ax(x), ay(y), az(z);
    ArrayInfo<int   ,1> ai(i), aj(j), ak(k);
    ArrayInfo<H_t   ,1> ah(h);
    ax.assert_identical_shape(ay);
    ax.assert_identical_shape(az);
    ax.assert_identical_shape(ai);
    ax.assert_identical_shape(aj);
    ax.assert_identical_shape(ak);
    ax.assert_identical_shape(ah);

    I_t const n = ax.shape(0);
    for( I_t ip=0; ip<n; ip++ ) {
        hilbert::xyzw2ijkh_float64( ax[ip], ay[ip], az[ip], w, ai[ip], aj[ip], ak[ip], ah[ip]);
    }
}

void xyzw2ijkh_float32
  ( py::array_t<float> x // in
  , py::array_t<float> y // in
  , py::array_t<float> z // in
  ,             float  w // in
  , py::array_t<int>   i // out
  , py::array_t<int>   j // out
  , py::array_t<int>   k // out
  , py::array_t<H_t>   h // out
  )
{
    ArrayInfo<float,1> ax(x), ay(y), az(z);
    ArrayInfo<int  ,1> ai(i), aj(j), ak(k);
    ArrayInfo<H_t  ,1> ah(h);
    ax.assert_identical_shape(ay);
    ax.assert_identical_shape(az);
    ax.assert_identical_shape(ai);
    ax.assert_identical_shape(aj);
    ax.assert_identical_shape(ak);
    ax.assert_identical_shape(ah);

    I_t const n = ax.shape(0);
    for( I_t ip=0; ip<n; ip++ ) {
        hilbert::xyzw2ijkh_float32( ax[ip], ay[ip], az[ip], w, ai[ip], aj[ip], ak[ip], ah[ip]);
    }
}

void rw2ch_float64
  ( py::array_t<double> r // in
  ,             double  w // in
  , py::array_t<int>    c // out
  , py::array_t<H_t>    h // out
  )
{
    ArrayInfo<double,2> ar(r);
    ArrayInfo<int   ,2> ac(c);
    ArrayInfo<H_t   ,1> ah(h);

    I_t const n = ar.shape(0);
    for( I_t ip=0; ip<n; ip++ ) {
        double const * p = &ar[3*ip];
        hilbert::rw2ch_float64( p, w, &ac[3*ip], ah[ip]);
    }
}

void rw2ch_float32
  ( py::array_t<float> r // in
  ,             float  w // in
  , py::array_t<int>   c // out
  , py::array_t<H_t>   h // out
  )
{
    ArrayInfo<float,2> ar(r);
    ArrayInfo<int  ,2> ac(c);
    ArrayInfo<H_t  ,1> ah(h);

    I_t const n = ar.shape(0);
    for( I_t ip=0; ip<n; ip++ ) {
        hilbert::rw2ch_float32( &ar[3*ip], w, &ac[3*ip], ah[ip]);
    }
}

void sort
  ( py::array_t<H_t> h
  , py::array_t<I_t> I
  )
{
    ArrayInfo<H_t,1> ah(h);
    ArrayInfo<I_t,1> aI(I);
    hilbert::insertion_sort( ah.shape(0), ah.data(), aI.data() );
}

void reorder_float64
  ( py::array_t<I_t> I
  , py::array_t<double> told
  , py::array_t<double> tnew
  )
{
    ArrayInfo<I_t   ,1> aI(I);
    ArrayInfo<double,1> atold(told), atnew(tnew);

    hilbert::reorder( aI.shape(0), aI.cdata(), atold.data(), atnew.data() );
}

void reorder2_float64
  ( py::array_t<I_t> I
  , py::array_t<double> told
  , py::array_t<double> tnew
  )
{// reorder 2-d array, e.g. r, v, a
    ArrayInfo<I_t   ,1> aI(I);
    ArrayInfo<double,2> atold(told), atnew(tnew);

    hilbert::reorder2( atold.shape(0), atold.shape(1)
                     , aI.cdata(), atold.data(), atnew.data() );
}

void reorder2_float32
  ( py::array_t<I_t  > I
  , py::array_t<float> told
  , py::array_t<float> tnew
  )
{// reorder 2-d array, e.g. r, v, a
    ArrayInfo<I_t  ,1> aI(I);
    ArrayInfo<float,2> atold(told), atnew(tnew);

    hilbert::reorder2( atold.shape(0), atold.shape(1)
                     , aI.cdata(), atold.data(), atnew.data() );
}

void reorder_float32
  ( py::array_t<I_t> I
  , py::array_t<float> told
  , py::array_t<float> tnew
  )
{
    ArrayInfo<I_t   ,1> aI(I);
    ArrayInfo<float,1> atold(told), atnew(tnew);

    hilbert::reorder( aI.shape(0), aI.cdata(), atold.data(), atnew.data() );
}

void reorder_int32
  ( py::array_t<I_t> I
  , py::array_t<int> told
  , py::array_t<int> tnew
  )
{
    ArrayInfo<I_t,1> aI(I);
    ArrayInfo<int,1> atold(told), atnew(tnew);

    hilbert::reorder( aI.shape(0), aI.cdata(), atold.data(), atnew.data() );
}

void reorder_uint32
  ( py::array_t<I_t> I
  , py::array_t<unsigned int> told
  , py::array_t<unsigned int> tnew
  )
{
    ArrayInfo<I_t,1> aI(I);
    ArrayInfo<unsigned int,1> atold(told), atnew(tnew);

    hilbert::reorder( aI.shape(0), aI.cdata(), atold.data(), atnew.data() );
}

void reorder_longlongint
  ( py::array_t<I_t> I
  , py::array_t<long long int> told
  , py::array_t<long long int> tnew
  )
{
    ArrayInfo<I_t,1> aI(I);
    ArrayInfo<long long int,1> atold(told), atnew(tnew);

    hilbert::reorder( aI.shape(0), aI.cdata(), atold.data(), atnew.data() );
}

inline H_t
ijk2h_1( py::array_t<int> ijk )
{
    ArrayInfo<int,1> aijk(ijk);

    int const *ptr_ijk = aijk.cdata();
    if( hilbert::is_valid_ijk(ptr_ijk) ) {
        H_t h = hilbert::ijk2h(ptr_ijk[0],ptr_ijk[1],ptr_ijk[2]);
        return h;
    } else {
        return -1;
    }
}

inline bool
h2ijk_1( H_t const h, py::array_t<int> ijk )
{
    ArrayInfo<int,1> aijk(ijk);
    int *ptr_ijk = aijk.data();
    if( hilbert::is_valid_h(h) ) {
        hilbert::h2ijk(h,aijk.data());
        return true;
    } else {
        return false;
    }
}

void
h2ijk( H_t const h, py::array_t<int> ijk )
{
    ArrayInfo<int,1> aijk(ijk);
    assert(aijk.shape(0)==3);
//    std::cout<<"h2ijk: h="<<h<<", ijk=["<<aijk[0]<<' '<<aijk[1]<<' '<<aijk[2]<<"]"<<std::endl;
    hilbert::h2ijk(h, aijk.data());
//    std::cout<<"h2ijk: h="<<h<<", ijk=["<<aijk[0]<<' '<<aijk[1]<<' '<<aijk[2]<<"] !!"<<std::endl;
}

void
build_hl
  ( py::array_t<H_t> h          // in
  , py::array_t<I_t> hl_offset  // out
  , py::array_t<I_t> hl_natoms  // out
  )
{
    ArrayInfo<H_t,1> a_h(h);
    ArrayInfo<I_t,1> a_hl_offset(hl_offset);
    ArrayInfo<I_t,1> a_hl_natoms(hl_natoms);

    std::size_t natoms = a_h.shape(0);
    std::size_t ncells = a_hl_offset.shape(0);
    std::size_t ia0 = 0;
    std::size_t ia  = 0;
    //std::cout<<"natoms="<<natoms<<std::endl;
    //std::cout<<"ncells="<<ncells<<std::endl;
    for( std::size_t hi=0; hi<ncells; ++hi )
    {
        //std::cout<<"hi="<<hi<<std::endl;
        while( ia < natoms && a_h[ia] == hi ) {
            ++ia;
            //std::cout<<"ia="<<ia<<std::endl;
        }
        a_hl_offset[hi] = ia0;
        a_hl_natoms[hi] = ia - ia0;
        ia0 = ia;
        //std::cout<<">>"<<a_hl_offset[hi]<<':'<<a_hl_natoms[hi]<<std::endl;
        if( ia >= natoms ) {
            break;
        }
    }
}

#include "../../../verletlist_cpp/vl_lib/vl_lib.hpp"
void
build_vl
  ( py::array_t<I_t    > K          // in
  , py::array_t<H_t    > h          // in
  , py::array_t<I_t    > hl_offset  // in
  , py::array_t<I_t    > hl_natoms  // in
  , py::array_t<double> r           // in
  , VL &                vl          //out
  )
{
    ArrayInfo<I_t,1> a_K(K);
    ArrayInfo<H_t,1> a_h(h);
    ArrayInfo<I_t,1> a_hl_offset(hl_offset);
    ArrayInfo<I_t,1> a_hl_natoms(hl_natoms);
    ArrayInfo<double,2> a_r(r);

    std::size_t natoms = a_h.shape(0);
    std::size_t ncells = a_hl_offset.shape(0);

    int ijk_central[3];
    int ijk_delta[13][3] = { { 1, 0, 0}  // x-direction
                           , {-1, 1, 0} // y-direction
                           , { 0, 1, 0}
                           , { 1, 1, 0}
                           , {-1,-1, 1} // z-direction
                           , { 0,-1, 1}
                           , { 1,-1, 1}
                           , {-1, 0, 1}
                           , { 0, 0, 1}
                           , { 1, 0, 1}
                           , {-1, 1, 1}
                           , { 0, 1, 1}
                           , { 1, 1, 1}
                           };
    int ijk_nb[13][3];
    double cutoff2 = vl.cutoff()*vl.cutoff();
    double ri [3];
    double rj [3];

 // Loop over all cells:
    H_t h_prev = std::numeric_limits<H_t>::max();

    for( std::size_t ih=0; ih<a_h.shape(0); ++ih )
    {
        H_t h_central = a_h[ih];
        if( h_central == h_prev ) {
            continue; // each cell must be visited only once
        } else {
            h_prev = h_central;
        }
     // compute the cell indices of central cell's
        hilbert::h2ijk(h_central, ijk_central);
//        std::cout<<" _central:"<<h_central<<" ijk_central["<<ijk_central[0]<<" "<<ijk_central[1]<<" "<<ijk_central[2]<<"]"<<std::endl;
     // compute the cell indices of all the neighbouring cells
        for( std::size_t i=0; i<13; ++i)
        for( std::size_t j=0; j< 3; ++j) {
            ijk_nb[i][j] = ijk_central[j] + ijk_delta[i][j];
        }
     // note that the ijk_delta stencil may refer to nonexisting cells (outside the box)

     // ijk_central-ijk_central pairs
        std::size_t offset_central = a_hl_offset[h_central];
        std::size_t natoms_central = a_hl_natoms[h_central];
        for( std::size_t i=offset_central; i<offset_central+natoms_central-1; ++i )
        {
            ri[0] = a_r[3*i];
            ri[1] = a_r[3*i+1];
            ri[2] = a_r[3*i+2];
            for( std::size_t j=i+1; j<offset_central+natoms_central; ++j )
            {
                rj[0] = a_r[3*j];
                rj[1] = a_r[3*j+1];
                rj[2] = a_r[3*j+2];
                double rij2 = 0.0;
                for( std::size_t k=0; k<3; ++k)
                    rij2 += std::pow( (rj[k] - ri[k]), 2 );
                if( rij2 <= cutoff2)
                    vl.add(i,j);
            }
        }

     // ijk_central-ijk_nb pairs
        for(std::size_t nb=0; nb<13; ++nb )
        {
            //std::cout<<"c-nb"<<nb<<" ijk_nb["<<ijk_nb[nb][0]<<' '<<ijk_nb[nb][1]<<' '<<ijk_nb[nb][2]<<"]"<<std::endl;
            if( -1 < ijk_nb[nb][0] && ijk_nb[nb][0] < a_K[0]
             && -1 < ijk_nb[nb][1] && ijk_nb[nb][1] < a_K[1]
             && -1 < ijk_nb[nb][2] && ijk_nb[nb][2] < a_K[2]
            ) // Otherwise the cell is outside the box
            {
                H_t h_nb = hilbert::ijk2h(ijk_nb[nb][0], ijk_nb[nb][1], ijk_nb[nb][2]);
                std::cout<<"h_nb:"<<h_nb<<" ijk_nb["<<ijk_nb[nb][0]<<' '<<ijk_nb[nb][1]<<' '<<ijk_nb[nb][2]<<"]"<<std::endl;
                std::size_t offset_nb = a_hl_offset[h_nb];
                std::size_t natoms_nb = a_hl_natoms[h_nb];
                for( std::size_t i = offset_central; i <offset_central + natoms_central; ++i)
                {
                    ri[0] = a_r[3*i];
                    ri[1] = a_r[3*i+1];
                    ri[2] = a_r[3*i+2];
                    for( std::size_t j = offset_nb; j < offset_nb + natoms_nb; ++j )
                    {
                        rj[0] = a_r[3*j];
                        rj[1] = a_r[3*j+1];
                        rj[2] = a_r[3*j+2];
                        double rij2 = 0.0;
                        for( std::size_t k=0; k<3; ++k)
                            rij2 += std::pow( (rj[k] - ri[k]), 2 );
                        if( rij2 <= cutoff2)
                            vl.add(i,j);
                    }
                }
            }
        }// end loop over neighbour cells
    }// end loop over cells
    vl.linearise(false);
}

PYBIND11_MODULE(spatialsorting, m)
{// optional module doc-string
    m.doc() = "pybind11 corecpp plugin"; // optional module docstring

    m.def("xyzw2h_dp"   , xyzw2h_float64 );    // 3D positions to hilbert index of the corresponding cell with width w.
    m.def("xyzw2h_sp"   , xyzw2h_float32 );    // 3D positions to hilbert index of the corresponding cell with width w.

    m.def("xyzw2ijkh_sp", xyzw2ijkh_float32 ); // 3D positions to cell indices and hilbert index of the corresponding cell with width w.
    m.def("xyzw2ijkh_dp", xyzw2ijkh_float64 ); // 3D positions to cell indices and hilbert index of the corresponding cell with width w.

    m.def(  "rw2h_dp"   ,   rw2h_float64 );    // 3D positions to hilbert index of the corresponding cell with width w.
    m.def(  "rw2h_sp"   ,   rw2h_float32 );    // 3D positions to hilbert index of the corresponding cell with width w.

    m.def("rw2ch_dp"    , rw2ch_float64 );     // 3D positions to cell indices and hilbert index of the corresponding cell with width w.
    m.def("rw2ch_sp"    , rw2ch_float32 );     // 3D positions to cell indices and hilbert index of the corresponding cell with width w.

    m.def("sort"               , sort   );     // sort the hilbert indices and produce reordering array.
    m.def("reorder_sp"    , reorder_float32);  // reorder single precision array
    m.def("reorder_dp"    , reorder_float64);  // reorder double precision array
    m.def("reorder_int32"      , reorder_int32  );// reorder int32 array
    m.def("reorder_uint32"     , reorder_uint32 );// reorder uint32 array
    m.def("reorder_longlongint", reorder_longlongint );// reorder longlongint
    m.def("reorder_2d_dp"   , reorder2_float64);// reorder 2D double precision array
    m.def("reorder_2d_sp"   , reorder2_float32);// reorder 2D single precision array

    m.def("ijk2h_1", ijk2h_1);
    m.def("h2ijk_1", h2ijk_1);
    m.def("h2ijk"  , h2ijk);

    m.def("info", hilbert::info);
    m.def("cell_index_limit"      , hilbert::cell_index_limit);
    m.def("hilbert_index_limit"   , hilbert::hilbert_index_limit);
    m.def("validate_cell_index"   , hilbert::validate_cell_index);
    m.def("validate_hilbert_index", hilbert::validate_hilbert_index);
    m.def("is_validating"         , hilbert::is_validating);
    m.def("is_valid_ijk"          , hilbert::is_valid_ijk);
    m.def("is_valid_i"            , hilbert::is_valid_i);
    m.def("is_valid_h"            , hilbert::is_valid_h);

    m.def("build_hl", build_hl);
    m.def("build_vl", build_vl);
}