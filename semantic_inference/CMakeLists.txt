cmake_minimum_required(VERSION 3.16)
project(semantic_inference VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(SEMANTIC_INFERENCE_USE_TRT "Build with TensorRT" ON)
option(SEMANTIC_INFERENCE_BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Build shared libs" ON)

find_package(config_utilities REQUIRED)
find_package(OpenCV REQUIRED)

find_package(CUDA)
find_package(TensorRT COMPONENTS onnxparser infer_plugin)

set(ENABLE_TENSORRT_CXX_VALUE 1)
if(SEMANTIC_INFERENCE_USE_TRT
   AND NOT CUDA_FOUND
   AND NOT TensorRT_FOUND
)
  message(WARNING "TensorRT is required for inference to run!"
                  " Follow the install instructions (if possible) to enable inference"
  )
  set(ENABLE_TENSORRT_CXX_VALUE 0)
elseif(NOT SEMANTIC_INFERENCE_USE_TRT)
  message(STATUS "Disabling TensorRT! Inference will not work!")
  set(ENABLE_TENSORRT_CXX_VALUE 0)
endif()

configure_file(
  cmake/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/semantic_inference_config.h
)

add_library(
  ${PROJECT_NAME}
  src/image_recolor.cpp src/image_rotator.cpp src/image_utilities.cpp src/logging.cpp
  src/model_config.cpp src/segmenter.cpp
)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC config_utilities::config_utilities ${OpenCV_LIBRARIES}
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${OpenCV_INCLUDE_DIRS}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(
  semantic_inference::${PROJECT_NAME} ALIAS ${PROJECT_NAME}
)

if(ENABLE_TENSORRT_CXX_VALUE)
  target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::TensorRT ${CUDA_LIBRARIES})
  target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
  target_sources(${PROJECT_NAME} PRIVATE src/trt_utilities.cpp src/model.cpp)
endif()

add_executable(demo_segmentation app/demo_segmentation.cpp)
target_link_libraries(demo_segmentation ${PROJECT_NAME})

if(SEMANTIC_INFERENCE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
  TARGETS ${PROJECT_NAME} demo_segmentation
  EXPORT ${PROJECT_NAME}-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
