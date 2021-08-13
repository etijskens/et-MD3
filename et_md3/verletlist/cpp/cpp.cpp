/*
 *  C++ source file for submodule et_md3.verletlist.cpp (binary extension)
 *
 *  Expose a VL class, implementing a Verlet list
 */


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "vl_lib/vl_lib.hpp"


PYBIND11_MODULE(cpp, m)
{// optional module doc-string
    m.doc() = "Expose the C++ implementation of Verlet list "; // optional module docstring
 // list the things you want to expose:
    py::class_<VL>(m, "VL")
        .def(py::init<std::size_t, double>())
        .def("reset"     , &VL::reset)
        .def("add"       , &VL::add)
        .def("linearise" , &VL::linearise)
        .def("natoms"    , &VL::natoms)
        .def("has"       , &VL::has)
        .def("print"     , &VL::print)
        .def("contact"   , &VL::contact)
        .def("ncontacts" , &VL::ncontacts)
        .def("cutoff"    , &VL::cutoff)
    ;
}
