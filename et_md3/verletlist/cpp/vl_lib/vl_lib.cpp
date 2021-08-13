/*
 *  C++ source file for module et_md.VL
 *
 *  Expose a VL class, implementing a Verlet list
 */


// See http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html for examples on how to use pybind11.
// The example below is modified after http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html#More-on-working-with-numpy-arrays
#include "vl_lib.hpp"

VL::VL
  ( std::size_t natoms
  , double cutoff
  )
  : cutoff_(cutoff)
{
    this->reset(natoms);
}

void
VL::reset( std::size_t n_atoms )
{
    this->vl2d_.resize(n_atoms);

    std::size_t ncontacts = std::min( n_atoms-1, (std::size_t)NCONTACTS );

    for( auto& itr : vl2d_ ) {
        if( itr.capacity() < ncontacts )
            itr.reserve(ncontacts);
    }
    this->linearised_ = false;
}

double
VL::cutoff() const {
    return cutoff_;
}

std::size_t
VL::natoms() const
{
    return ( linearised_ ? vl_offset_.size() : vl2d_.size() );
}

void
VL::add(std::size_t i, std::size_t j)
{
  #ifdef DEBUG
    validate_atom(i);
    validate_atom(j);
    // If we are not validating, and the indices are out of range, you get
    // MemoryError: std::bad_alloc
  #endif

    if( this->linearised_ )
        throw std::runtime_error("Cannot add to linearised Verlet list.");
    vl2d_[i].push_back(j);
}

// range check for atom i
void
VL::validate_atom(std::size_t i) const
{
    if( i >= this->natoms() ) {
        std::string msg("No such atom: ");
        msg += std::to_string(i) + ", valid range is ["
             + std::to_string(0) + "," + std::to_string(this->natoms()) + "[.";
        throw std::runtime_error(msg);
    }
}

bool
VL::has(std::size_t i, std::size_t j) const
{
    validate_atom(i);
    validate_atom(j);

    if( linearised_ ) {
        for( std::size_t k=vl_offset_[i]; k<vl_offset_[i]+vl_natoms_[i]; ++k ) {
            if( vl_[k] == j )
                return true;
        }
    } else {
        for( auto& itr : vl2d_[i] ) {
            if( itr == j )
                return true;
        }
    }
    return false;
}

void
VL::print() const
{
    if( linearised_ ) {
        std::cout<<"VL::print() (linearised):\n";
        for(std::size_t i=0; i<natoms(); ++i) {
            std::cout<<i<<" [";
            for( std::size_t k=vl_offset_[i]; k<vl_offset_[i]+vl_natoms_[i]; ++k ) {
                std::cout<<vl_[k]<<",";
            }
            std::cout<<"]"<<std::endl;
        }
    } else {
        std::cout<<"VL::print():\n";
        for(std::size_t i=0; i<natoms(); ++i) {
            std::cout<<i<<" [";
            for( auto & ktr : vl2d_[i] )
                std::cout<<ktr<<",";
            std::cout<<"]"<<std::endl;
        }
    }
}

void
VL::linearise(bool keep2d)
{
    std::size_t natoms = vl2d_.size();
    vl_offset_.resize(natoms);
    vl_natoms_.resize(natoms);

    std::size_t ncontacts = 0;
    for( auto& itr : vl2d_ ) {
        ncontacts += itr.size();
    }
//        std::cout<<"ncontacts = "<<ncontacts<<std::endl;
    vl_.reserve(ncontacts);
    std::size_t offset = 0;
    std::size_t i = 0;
    for( auto& itr : vl2d_ ) {
        vl_offset_[i] = offset;
        vl_natoms_[i] = itr.size();
        offset       += itr.size();
        vl_.insert( vl_.end(), itr.begin(), itr.end() );
        i += 1;
    }
    linearised_ = true;
//        for( auto& ij : vl_) std::cout<<ij<<" ";
//        std::cout<<std::endl;
    if( !keep2d ) {
        vl2d_.clear();
    }
}

std::size_t
VL::ncontacts( std::size_t i ) const
{
    if( linearised_ ) {
        return vl_natoms_[i];
    } else{
        return vl2d_[i].size();
    }
}

std::size_t
VL::contact( std::size_t i, std::size_t j) const
{
    if( linearised_ ) {
        return vl_[vl_offset_[i] + j ];
    } else{
        return vl2d_[i][j];
    }
}
