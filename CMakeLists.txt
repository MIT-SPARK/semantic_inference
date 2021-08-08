cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor_nodelet)

# TODO(nathan) compile options
find_package(catkin REQUIRED cv_bridge)

catkin_package(CATKIN_DEPENDS cv_bridge DEPENDS INCLUDE_DIRS include LIBRARIES)
