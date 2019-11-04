project(SubnormalTest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2")
add_executable(subnormal_test subnormal_test.cpp)
