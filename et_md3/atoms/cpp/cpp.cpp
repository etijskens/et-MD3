/*
 *  C++ source file for module et_md3.atoms.cpp
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "ArrayInfo.hpp"

template <typename FloatType>
void
scale_forces
  ( py::array_t<FloatType> a // forces
  , py::array_t<FloatType> m // masses
  )
{
    ArrayInfo<FloatType,2> a__(a);
    ArrayInfo<FloatType,1> m__(m);
    std::size_t const n = a__.shape(0);
    FloatType      * pa = &a__[0];
    FloatType const* pm = &m__[0];
    for( std::size_t i=0; i<n; ++i, ++pm) {
        for( std::size_t k=0; k<3; ++k, ++pa) {
            *pa /= *pm;
        }
    }
}


PYBIND11_MODULE(cpp, m)
{// optional module doc-string
    m.doc() = "pybind11 cpp plugin"; // optional module docstring
 // list the functions you want to expose:
 // m.def("exposed_name", function_pointer, "doc-string for the exposed function");
    m.def("scale_forces_sp", &scale_forces<float>, " a[:,k] /= m, for k=0,1,2.");
    m.def("scale_forces_dp", &scale_forces<double>, " a[:,k] /= m, for k=0,1,2.");
}
