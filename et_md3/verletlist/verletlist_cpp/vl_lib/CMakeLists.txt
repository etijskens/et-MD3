cmake_minimum_required(VERSION 3.4)

# vllib is a shared library so it can be linked to binary extension modules
# that want to use it from C++, rather than from python.

message("SUBDIRECTORY vl_lib")

project(vl_lib CXX)

set(CMAKE_CXX_STANDARD 17)

add_library( vl_lib
  SHARED
# source file(s)
  vl_lib.cpp
)

# The shared library is installed at the same location as the binary extension modules 
install(TARGETS vl_lib
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)