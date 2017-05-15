# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/eliezer/software/clion-2016.2.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/eliezer/software/clion-2016.2.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/eliezer/Dropbox/repo-phd/poissoncpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/eliezer/Dropbox/repo-phd/poissoncpp

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/home/eliezer/software/clion-2016.2.3/bin/cmake/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/eliezer/software/clion-2016.2.3/bin/cmake/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/eliezer/Dropbox/repo-phd/poissoncpp/CMakeFiles /home/eliezer/Dropbox/repo-phd/poissoncpp/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/eliezer/Dropbox/repo-phd/poissoncpp/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named poisson_weighted_learn_hyper

# Build rule for target.
poisson_weighted_learn_hyper: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 poisson_weighted_learn_hyper
.PHONY : poisson_weighted_learn_hyper

# fast build rule for target.
poisson_weighted_learn_hyper/fast:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/build
.PHONY : poisson_weighted_learn_hyper/fast

#=============================================================================
# Target rules for targets named poisson_weighted_learn

# Build rule for target.
poisson_weighted_learn: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 poisson_weighted_learn
.PHONY : poisson_weighted_learn

# fast build rule for target.
poisson_weighted_learn/fast:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/build
.PHONY : poisson_weighted_learn/fast

#=============================================================================
# Target rules for targets named poisson_scr_cpp

# Build rule for target.
poisson_scr_cpp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 poisson_scr_cpp
.PHONY : poisson_scr_cpp

# fast build rule for target.
poisson_scr_cpp/fast:
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/build
.PHONY : poisson_scr_cpp/fast

BatchPoissonPure.o: BatchPoissonPure.cpp.o

.PHONY : BatchPoissonPure.o

# target to build an object file
BatchPoissonPure.cpp.o:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/BatchPoissonPure.cpp.o
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/BatchPoissonPure.cpp.o
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/BatchPoissonPure.cpp.o
.PHONY : BatchPoissonPure.cpp.o

BatchPoissonPure.i: BatchPoissonPure.cpp.i

.PHONY : BatchPoissonPure.i

# target to preprocess a source file
BatchPoissonPure.cpp.i:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/BatchPoissonPure.cpp.i
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/BatchPoissonPure.cpp.i
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/BatchPoissonPure.cpp.i
.PHONY : BatchPoissonPure.cpp.i

BatchPoissonPure.s: BatchPoissonPure.cpp.s

.PHONY : BatchPoissonPure.s

# target to generate assembly for a file
BatchPoissonPure.cpp.s:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/BatchPoissonPure.cpp.s
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/BatchPoissonPure.cpp.s
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/BatchPoissonPure.cpp.s
.PHONY : BatchPoissonPure.cpp.s

BatchPoissonWeight.o: BatchPoissonWeight.cpp.o

.PHONY : BatchPoissonWeight.o

# target to build an object file
BatchPoissonWeight.cpp.o:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/BatchPoissonWeight.cpp.o
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/BatchPoissonWeight.cpp.o
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/BatchPoissonWeight.cpp.o
.PHONY : BatchPoissonWeight.cpp.o

BatchPoissonWeight.i: BatchPoissonWeight.cpp.i

.PHONY : BatchPoissonWeight.i

# target to preprocess a source file
BatchPoissonWeight.cpp.i:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/BatchPoissonWeight.cpp.i
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/BatchPoissonWeight.cpp.i
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/BatchPoissonWeight.cpp.i
.PHONY : BatchPoissonWeight.cpp.i

BatchPoissonWeight.s: BatchPoissonWeight.cpp.s

.PHONY : BatchPoissonWeight.s

# target to generate assembly for a file
BatchPoissonWeight.cpp.s:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/BatchPoissonWeight.cpp.s
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/BatchPoissonWeight.cpp.s
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/BatchPoissonWeight.cpp.s
.PHONY : BatchPoissonWeight.cpp.s

datasets.o: datasets.cpp.o

.PHONY : datasets.o

# target to build an object file
datasets.cpp.o:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/datasets.cpp.o
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/datasets.cpp.o
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/datasets.cpp.o
.PHONY : datasets.cpp.o

datasets.i: datasets.cpp.i

.PHONY : datasets.i

# target to preprocess a source file
datasets.cpp.i:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/datasets.cpp.i
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/datasets.cpp.i
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/datasets.cpp.i
.PHONY : datasets.cpp.i

datasets.s: datasets.cpp.s

.PHONY : datasets.s

# target to generate assembly for a file
datasets.cpp.s:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/datasets.cpp.s
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/datasets.cpp.s
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/datasets.cpp.s
.PHONY : datasets.cpp.s

main-learn.o: main-learn.cpp.o

.PHONY : main-learn.o

# target to build an object file
main-learn.cpp.o:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/main-learn.cpp.o
.PHONY : main-learn.cpp.o

main-learn.i: main-learn.cpp.i

.PHONY : main-learn.i

# target to preprocess a source file
main-learn.cpp.i:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/main-learn.cpp.i
.PHONY : main-learn.cpp.i

main-learn.s: main-learn.cpp.s

.PHONY : main-learn.s

# target to generate assembly for a file
main-learn.cpp.s:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn.dir/build.make CMakeFiles/poisson_weighted_learn.dir/main-learn.cpp.s
.PHONY : main-learn.cpp.s

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/poisson_scr_cpp.dir/build.make CMakeFiles/poisson_scr_cpp.dir/main.cpp.s
.PHONY : main.cpp.s

main_hyper.o: main_hyper.cpp.o

.PHONY : main_hyper.o

# target to build an object file
main_hyper.cpp.o:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/main_hyper.cpp.o
.PHONY : main_hyper.cpp.o

main_hyper.i: main_hyper.cpp.i

.PHONY : main_hyper.i

# target to preprocess a source file
main_hyper.cpp.i:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/main_hyper.cpp.i
.PHONY : main_hyper.cpp.i

main_hyper.s: main_hyper.cpp.s

.PHONY : main_hyper.s

# target to generate assembly for a file
main_hyper.cpp.s:
	$(MAKE) -f CMakeFiles/poisson_weighted_learn_hyper.dir/build.make CMakeFiles/poisson_weighted_learn_hyper.dir/main_hyper.cpp.s
.PHONY : main_hyper.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... poisson_weighted_learn_hyper"
	@echo "... poisson_weighted_learn"
	@echo "... poisson_scr_cpp"
	@echo "... BatchPoissonPure.o"
	@echo "... BatchPoissonPure.i"
	@echo "... BatchPoissonPure.s"
	@echo "... BatchPoissonWeight.o"
	@echo "... BatchPoissonWeight.i"
	@echo "... BatchPoissonWeight.s"
	@echo "... datasets.o"
	@echo "... datasets.i"
	@echo "... datasets.s"
	@echo "... main-learn.o"
	@echo "... main-learn.i"
	@echo "... main-learn.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... main_hyper.o"
	@echo "... main_hyper.i"
	@echo "... main_hyper.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

