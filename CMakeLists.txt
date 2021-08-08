cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor_nodelet)

# TODO(nathan) compile options
find_package(catkin REQUIRED cv_bridge)

# download libtocch
set(PYTORCH_CACHE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_cache")
set(PYTORCH_OUTPUT_DIR "${CMAKE_BINARY_DIR}/pytorch-download/libtorch")
configure_file(cmake/pytorch.CMakeLists.txt.in pytorch-download/CMakeLists.txt)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch-download"
  RESULT_VARIABLE PYTORCH_CONFIGURE_RET)
if(NOT PYTORCH_CONFIGURE_RET EQUAL "0")
  message(FATAL_ERROR "pytorch configure step failed!")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch-download"
  RESULT_VARIABLE PYTORCH_BUILD_RET)
if(NOT PYTORCH_BUILD_RET EQUAL "0")
  message(FATAL_ERROR "pytorch configure step failed!")
endif()

# download and build specific protobuf version for libtorch
set(PROTOBUF_INSTALL_DIR "${CMAKE_BINARY_DIR}/protobuf-build")
configure_file(cmake/protobuf.CMakeLists.txt.in protobuf-download/CMakeLists.txt)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/protobuf-download"
  RESULT_VARIABLE PROTOBUF_CONFIGURE_RET)
if(NOT PROTOBUF_CONFIGURE_RET EQUAL "0")
  message(FATAL_ERROR "protobuf configure step failed!")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build . -- -j
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/protobuf-download"
  RESULT_VARIABLE PROTOBUF_BUILD_RET)
if(NOT PROTOBUF_BUILD_RET EQUAL "0")
  message(FATAL_ERROR "protobuf build step failed!")
endif()

list(
  APPEND
  CMAKE_PREFIX_PATH
  "${CMAKE_BINARY_DIR}/pytorch-download/libtorch/libtorch/share/cmake"
)
find_package(Torch REQUIRED)

find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${CMAKE_BINARY_DIR}/protobuf-build/lib/pkgconfig")
pkg_check_modules(protobuf REQUIRED protobuf=3.11.4 IMPORTED_TARGET)

#list(
  #APPEND
  #CMAKE_PREFIX_PATH
  #"${CMAKE_BINARY_DIR}/protobuf-download/pytorch_binaries-prefix/src/pytorch_binaries/share/cmake"
#)
#find_package(Protobuf REQUIRED)

catkin_package(CATKIN_DEPENDS cv_bridge DEPENDS INCLUDE_DIRS include LIBRARIES)

add_executable(test_caffe2 src/caffe2_mask_rcnn.cpp)
target_link_libraries(test_caffe2 ${catkin_LIBRARIES} "${TORCH_LIBRARIES}" PkgConfig::protobuf)
