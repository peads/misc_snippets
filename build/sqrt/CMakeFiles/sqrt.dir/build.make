# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/peads/misc_snippets

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/peads/misc_snippets/build

# Include any dependencies generated for this target.
include sqrt/CMakeFiles/sqrt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include sqrt/CMakeFiles/sqrt.dir/compiler_depend.make

# Include the progress variables for this target.
include sqrt/CMakeFiles/sqrt.dir/progress.make

# Include the compile flags for this target's objects.
include sqrt/CMakeFiles/sqrt.dir/flags.make

sqrt/CMakeFiles/sqrt.dir/sqrttests.c.o: sqrt/CMakeFiles/sqrt.dir/flags.make
sqrt/CMakeFiles/sqrt.dir/sqrttests.c.o: ../sqrt/sqrttests.c
sqrt/CMakeFiles/sqrt.dir/sqrttests.c.o: sqrt/CMakeFiles/sqrt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/peads/misc_snippets/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object sqrt/CMakeFiles/sqrt.dir/sqrttests.c.o"
	cd /home/peads/misc_snippets/build/sqrt && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT sqrt/CMakeFiles/sqrt.dir/sqrttests.c.o -MF CMakeFiles/sqrt.dir/sqrttests.c.o.d -o CMakeFiles/sqrt.dir/sqrttests.c.o -c /home/peads/misc_snippets/sqrt/sqrttests.c

sqrt/CMakeFiles/sqrt.dir/sqrttests.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sqrt.dir/sqrttests.c.i"
	cd /home/peads/misc_snippets/build/sqrt && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/peads/misc_snippets/sqrt/sqrttests.c > CMakeFiles/sqrt.dir/sqrttests.c.i

sqrt/CMakeFiles/sqrt.dir/sqrttests.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sqrt.dir/sqrttests.c.s"
	cd /home/peads/misc_snippets/build/sqrt && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/peads/misc_snippets/sqrt/sqrttests.c -o CMakeFiles/sqrt.dir/sqrttests.c.s

# Object files for target sqrt
sqrt_OBJECTS = \
"CMakeFiles/sqrt.dir/sqrttests.c.o"

# External object files for target sqrt
sqrt_EXTERNAL_OBJECTS =

sqrt/sqrt: sqrt/CMakeFiles/sqrt.dir/sqrttests.c.o
sqrt/sqrt: sqrt/CMakeFiles/sqrt.dir/build.make
sqrt/sqrt: sqrt/CMakeFiles/sqrt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/peads/misc_snippets/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable sqrt"
	cd /home/peads/misc_snippets/build/sqrt && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sqrt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sqrt/CMakeFiles/sqrt.dir/build: sqrt/sqrt
.PHONY : sqrt/CMakeFiles/sqrt.dir/build

sqrt/CMakeFiles/sqrt.dir/clean:
	cd /home/peads/misc_snippets/build/sqrt && $(CMAKE_COMMAND) -P CMakeFiles/sqrt.dir/cmake_clean.cmake
.PHONY : sqrt/CMakeFiles/sqrt.dir/clean

sqrt/CMakeFiles/sqrt.dir/depend:
	cd /home/peads/misc_snippets/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/peads/misc_snippets /home/peads/misc_snippets/sqrt /home/peads/misc_snippets/build /home/peads/misc_snippets/build/sqrt /home/peads/misc_snippets/build/sqrt/CMakeFiles/sqrt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sqrt/CMakeFiles/sqrt.dir/depend

