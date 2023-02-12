cmake_minimum_required(VERSION 3.22)
SET(PROJECT_NAME inverse_sqrt)
project(${PROJECT_NAME} LANGUAGES C)
set(CMAKE_C_STANDARD 99)

FILE(GLOB SRCS "*.c")

add_executable(${PROJECT_NAME} ${SRCS})
