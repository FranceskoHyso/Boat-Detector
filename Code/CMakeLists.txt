cmake_minimum_required(VERSION 3.2)
project(boatDetector)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

#Where to find CMake modules and OpenCV
include_directories("C:\\\\opencvLibrary\\\\opencv4.5.1\\\\build\\\\install\\\\include\\\\opencv2")

add_executable(boatDetector boatDetector.cpp SupportClasses/Codebook.cpp SupportClasses/Codebook.h SupportClasses/Graph.cpp SupportClasses/Graph.h)

target_link_libraries(boatDetector  ${OpenCV_LIBS})