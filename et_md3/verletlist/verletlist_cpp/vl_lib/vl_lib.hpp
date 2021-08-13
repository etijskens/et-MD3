/*
 *  C++ header file for shared library vllib
 */

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

typedef  unsigned int I_t;

#define NCONTACTS 50
 //------------------------------------------------------------------------------
 // Basic Verlet list data structure, without a method build it.
    class VL
    {//------------------------------------------------------------------------------
        bool linearised_;
        double cutoff_;
     // 2d Verlet list
        std::vector< std::vector<std::size_t> > vl2d_;
     // linearized Verlet list
        std::vector< std::size_t > vl_;
        std::vector< std::size_t > vl_offset_;
        std::vector< std::size_t > vl_natoms_;

      public:
     // ctor
        VL( double cutoff );

     // Return the cutoff distance.
        double cutoff() const;

     // Return the number of atoms in the VL
        std::size_t natoms() const;

        std::size_t ncontacts( std::size_t i ) const;
        std::size_t contact( std::size_t i, std::size_t j ) const;

     // Check if pair (i,j) is in the VL.
        bool has(std::size_t i, std::size_t j) const;

     // Print the Verlet list of each atom to stdout.
        void print() const;

     // ----------------------------------------------------------------------------
     // Methods for vlbuilders
     // (re-)Allocate 2d data structure
        void allocate_2d( std::size_t n_atoms );

     // Add a contact (i,j), atom j is added to the Verlet list of atom i.
        void add(std::size_t i, std::size_t j);

     // Linearize the Verlet lists.
        void linearise( bool keep2d );

     // Throw std::runtime_error if atom i is outside the atom range.
        void validate_atom(std::size_t i) const;
    };
