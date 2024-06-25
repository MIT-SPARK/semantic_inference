find_package(GTest REQUIRED)
include(GoogleTest)
enable_testing()

add_executable(test_${PROJECT_NAME} utest_image_utilities.cpp utest_main.cpp)
target_link_libraries(test_${PROJECT_NAME} PRIVATE ${PROJECT_NAME} GTest::gtest_main)
gtest_add_tests(TARGET test_${PROJECT_NAME})
