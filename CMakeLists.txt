cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor_nodelet)

#TODO(nathan) compile options
find_package(catkin REQUIRED)


set(PROTOBUF_VERSION "3.11.4")
set(PROTOBUF_BASE_URL "https://github.com/protocolbuffers/protobuf/releases/download")
set(PROTOBUF_FILE "protobuf-cpp-${PROTOBUF_VERSION}.tar.gz")

# adapted from https://github.com/ethz-asl/protobuf_catkin
include(ExternalProject)
ExternalProject_Add(
  protobuf_src
  URL ${PROTOBUF_BASE_URL}/v${PROTOBUF_VERSION}/${PROTOBUF_FILE}
  CONFIGURE_COMMAND ./configure --prefix=${CATKIN_DEVEL_PREFIX}
  BUILD_IN_SOURCE TRUE
)

catkin_package()
