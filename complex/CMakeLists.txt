cmake_minimum_required(VERSION 3.22)
SET(PROJECT_NAME complex)
project(${PROJECT_NAME} LANGUAGES C)
set(CMAKE_C_STANDARD 99)

FILE(GLOB SRCS "*.c")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
include_directories("${CMAKE_HOME_DIRECTORY}/timed_functions/include")

add_compile_options(-Wno-argument-outside-range)
add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC timed_functions)