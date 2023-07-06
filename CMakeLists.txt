cmake_minimum_required(VERSION 3.14)
project(semantic_recolor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED roscpp nodelet cv_bridge image_transport)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glog_catkin REQUIRED)

configure_file(cmake/ort.CMakeLists.txt.in ort-download/CMakeLists.txt @ONLY)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ort-download" # OUTPUT_QUIET
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build .
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ort-download" # OUTPUT_QUIET
)

add_library(
  ort::ort IMPORTED INTERFACE
)
set_property(
  TARGET ort::ort PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                           "${CMAKE_CURRENT_BINARY_DIR}/ort-src/include"
)
set_property(
  TARGET ort::ort PROPERTY INTERFACE_LINK_LIBRARIES
                           "${CMAKE_CURRENT_BINARY_DIR}/ort-src/lib/libonnxruntime.so"
)

catkin_package(
  CATKIN_DEPENDS roscpp nodelet cv_bridge image_transport
  DEPENDS OpenCV
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

add_library(
  ${PROJECT_NAME}
  src/image_utilities.cpp
  src/model_config.cpp
  # src/rgbd_segmenter.cpp
  src/ros_utilities.cpp
  src/segmenter.cpp
  src/semantic_color_config.cpp
  src/yaml_utilities.cpp
  src/backends/ort/ort_backend.cpp
  src/backends/ort/ort_utilities.cpp
)
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp
  PRIVATE ort::ort ${glog_catkin_LIBRARIES}
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${glog_catkin_INCLUDE_DIRS}
)

add_library(
  ${PROJECT_NAME}_nodelet
  src/mask_nodelet.cpp src/nodelet_output_publisher.cpp src/segmentation_nodelet.cpp
  src/recolor_nodelet.cpp
)
target_include_directories(
  ${PROJECT_NAME}_nodelet PUBLIC include ${catkin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_nodelet ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(demo_segmentation src/demo_segmentation.cpp)
target_link_libraries(demo_segmentation ${PROJECT_NAME})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(
    utest_${PROJECT_NAME} tests/utest_main.cpp tests/utest_config.cpp
                          tests/utest_image_utilities.cpp
  )
  target_link_libraries(utest_${PROJECT_NAME} ${PROJECT_NAME})
endif()
