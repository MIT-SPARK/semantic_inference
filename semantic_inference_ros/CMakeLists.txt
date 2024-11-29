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
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(semantic_inference REQUIRED)

# SHARED is required for composable nodes
add_library(
  ${PROJECT_NAME} SHARED
  src/mask_nodelet.cpp src/rgbd_segmentation_nodelet.cpp src/segmentation_nodelet.cpp
  src/output_publisher.cpp src/recolor_nodelet.cpp
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                         "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC cv_bridge::cv_bridge image_transport::image_transport
         message_filters::message_filters rclcpp::rclcpp rclcpp_components::component
         semantic_inference::semantic_inference
)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN semantic_inference::MaskNode EXECUTABLE mask_node
)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN semantic_inference_ros::RecolorNode EXECUTABLE recolor_node
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  LIBRARY DESTINATION lib
)
install(PROGRAMS app/clip_publisher_node app/language_embedding_node app/make_rosbag
                 app/merge_rosbag app/openset_segmentation_node
        DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include/${PROJECT_NAME}/)
install(DIRECTORY launch/ DESTINATION shared/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(${PROJECT_NAME}-targets)
ament_python_install_package(${PROJECT_NAME})
ament_package()
