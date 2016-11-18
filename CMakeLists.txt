cmake_minimum_required(VERSION 3.6)
project(poisson_scr_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp BatchPoissonPure.h BatchPoissonPure.cpp)
add_executable(poisson_scr_cpp ${SOURCE_FILES} BatchPoissonPure.h BatchPoissonPure.cpp)