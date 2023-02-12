cmake_minimum_required(VERSION 3.22)
set(PROJECT_NAME timed_functions)
project(${PROJECT_NAME} LANGUAGES C)
set(CMAKE_C_STANDARD 99)

FILE(GLOB SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
include_directories("${CMAKE_HOME_DIRECTORY}/timed_functions/include")

add_library(${PROJECT_NAME} "${SRC}")
