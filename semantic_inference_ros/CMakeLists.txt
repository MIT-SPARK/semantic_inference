cmake_minimum_required(VERSION 3.16)
project(semantic_inference_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(CLI11 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ianvs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(message_filters REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosbag2_transport REQUIRED)
find_package(semantic_inference REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)

# SHARED is required for composable nodes
add_library(
  ${PROJECT_NAME} SHARED
  src/backprojection_nodelet.cpp
  src/mask_nodelet.cpp
  src/rgbd_segmentation_nodelet.cpp
  src/segmentation_nodelet.cpp
  src/output_publisher.cpp
  src/pointcloud_projection.cpp
  src/relabel_nodelet.cpp
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                         "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC rclcpp::rclcpp semantic_inference::semantic_inference
  PRIVATE cv_bridge::cv_bridge
          ianvs::ianvs
          image_geometry::image_geometry
          message_filters::message_filters
          nlohmann_json::nlohmann_json
          rclcpp_components::component
          tf2_eigen::tf2_eigen
          tf2_ros::tf2_ros
)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN semantic_inference::BackprojectionNode EXECUTABLE backprojection_node
)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN semantic_inference::SegmentationNode EXECUTABLE closed_set_node
)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN semantic_inference::MaskNode EXECUTABLE mask_node
)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN semantic_inference::RelabelNode EXECUTABLE relabel_node
)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN semantic_inference::RGBDSegmentationNode EXECUTABLE rgbd_closed_set_node
)

add_executable(closed_set_rosbag_writer app/closed_set_rosbag_writer.cpp)
target_link_libraries(
  closed_set_rosbag_writer PUBLIC ${PROJECT_NAME} cv_bridge::cv_bridge ianvs::ianvs_rosbag
                                  rosbag2_transport::rosbag2_transport CLI11::CLI11
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS app/image_embedding_node app/open_set_node app/text_embedding_node
        DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS closed_set_rosbag_writer RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include/${PROJECT_NAME}/)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # NOTE(nathan) this is janky because pcl, pcl_conversions, and ament can't do cmake right
  set(PCL_FIND_QUIETLY ON)
  find_package(PCL REQUIRED COMPONENTS common)
  find_package(pcl_conversions REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_test test/main.cpp test/test_pointcloud_projection.cpp)
  target_include_directories(${PROJECT_NAME}_test PRIVATE ${pcl_conversions_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${pcl_conversions_LIBRARIES})
endif()

ament_export_targets(${PROJECT_NAME}-targets)
ament_export_dependencies(rclcpp semantic_inference)
ament_python_install_package(${PROJECT_NAME})
ament_package()
