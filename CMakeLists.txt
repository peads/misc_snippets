cmake_minimum_required(VERSION 3.22)
SET(PROJECT_NAME misc_snippets)
project(${PROJECT_NAME} LANGUAGES C)
set(CMAKE_C_STANDARD 99)

#FILE(GLOB SRCS "src /*.c")
set(CMAKE_C_STANDARD 99)

add_subdirectory(inverse_sqrt)
add_subdirectory(sqrt)
add_subdirectory(complex)
add_subdirectory(timed_functions)

target_link_libraries(complex PUBLIC timed_functions)