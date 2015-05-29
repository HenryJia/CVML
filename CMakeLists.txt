cmake_minimum_required(VERSION 2.8.11)

project(CVML)



include_directories(
    /home/henry/Coding/C++/CVML/
    /home/henry/Coding/C++/CVML/

)

# Linker options
set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS}  ")

# Library path
set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS} -L\".\" ")

# Define the CXX sources
set ( CXX_SRCS
    main.cpp
    cvnn.cpp
)

set_source_files_properties(
    ${CXX_SRCS} PROPERTIES COMPILE_FLAGS 
    " -g -O1 -O -O3 -O0 -O2 -std=c++11 -Wall -I /usr/include/opencv/")

add_executable(CVML ${CXX_SRCS} ${C_SRCS})

target_link_libraries(CVML
    opencv_core
    opencv_features2d
    opencv_flann
    opencv_gpu
    opencv_highgui
    opencv_imgproc
    opencv_legacy
    opencv_ml
    opencv_nonfree
    opencv_objdetect
    opencv_ocl
    opencv_photo
    opencv_stitching
    opencv_superres
    opencv_ts
    opencv_video
    opencv_videostab
    tbb
    rt
    pthread
    m
    dl
)

