cmake_minimum_required(VERSION 3.6)
project(poisson_scr_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp  BatchPoissonPure.h BatchPoissonPure.cpp BatchPoissonWeight.h BatchPoissonWeight.cpp datasets.h datasets.cpp)
add_executable(poisson_scr_cpp ${SOURCE_FILES} BatchPoissonPure.h BatchPoissonPure.cpp BatchPoissonWeight.h BatchPoissonWeight.cpp  datasets.h datasets.cpp)

set(SOURCE_FILES main-learn.cpp BatchPoissonWeight.h BatchPoissonWeight.cpp datasets.h datasets.cpp BatchPoissonPure.h BatchPoissonPure.cpp)
add_executable(poisson_weighted_learn ${SOURCE_FILES} BatchPoissonWeight.h BatchPoissonWeight.cpp BatchPoissonPure.h BatchPoissonPure.cpp datasets.h datasets.cpp)

set(SOURCE_FILES main_hyper.cpp BatchPoissonWeight.h BatchPoissonWeight.cpp datasets.h datasets.cpp BatchPoissonPure.h BatchPoissonPure.cpp )
add_executable(poisson_weighted_learn_hyper ${SOURCE_FILES} BatchPoissonWeight.h BatchPoissonWeight.cpp BatchPoissonPure.h BatchPoissonPure.cpp datasets.h datasets.cpp )
