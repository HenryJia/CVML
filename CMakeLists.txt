cmake_minimum_required(VERSION 2.8.11)

project(CVML)



include_directories(
    /home/henry/Coding/C++/CVML/
    /home/henry/Coding/C++/CVML/

)

# Linker options
set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS}  ")

# Library path
set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS} -L\".\" -L\"OpenCV\" ")

# Define the CXX sources
set ( CXX_SRCS
    main.cpp
    cvnn.cpp
)

set_source_files_properties(
    ${CXX_SRCS} PROPERTIES COMPILE_FLAGS 
    " -g -O1 -O -O3 -O0 -O2 -std=c++11 -Wall -I /usr/include/opencv/")

find_package( OpenCV REQUIRED )

add_executable(CVML ${CXX_SRCS} ${C_SRCS})

target_link_libraries(CVML
    ${OpenCV_LIBS}
    tbb
    rt
    pthread
    m
    dl
)

