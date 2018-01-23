cmake_minimum_required(VERSION 2.8.3)
project(opencl_caffe_test)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
set(Caffe_INCLUDE_DIRS /home/ryaadhav/caffe/include;/home/ryaadhav/caffe/build/include)
set(Caffe_LIBS /home/ryaadhav/caffe/build/lib/libcaffe.so)

#find_package(Boost)


add_definitions( -DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" )
add_definitions( -DUSE_GREENTEA )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${${OpenCL_INCLUDE_DIRS}}
  ${Caffe_INCLUDE_DIRS}
  #${Boost_INCLUDE_DIRS}
)

add_executable(test test.cpp clManager.hpp clManager.cpp)
target_link_libraries(
  test
  ${OpenCL_LIBRARIES}
  ${Caffe_LIBS}
   ${OpenCV_LIBRARIES}
  -lboost_system
)
