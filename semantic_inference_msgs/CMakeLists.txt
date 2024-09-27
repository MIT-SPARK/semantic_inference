cmake_minimum_required(VERSION 3.16)
project(semantic_inference_msgs)

find_package(catkin REQUIRED message_generation sensor_msgs std_msgs)

add_message_files(
  FILES FeatureVector.msg FeatureVectors.msg FeatureVectorStamped.msg FeatureImage.msg
)
add_service_files(FILES EncodeFeature.srv)

generate_messages(DEPENDENCIES sensor_msgs std_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime sensor_msgs std_msgs
  DEPENDS
  INCLUDE_DIRS
  LIBRARIES
)
