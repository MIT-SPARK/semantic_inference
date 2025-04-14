#include <gtest/gtest.h>
#include <semantic_inference/logging.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  logging::Logger::addSink("stdout", std::make_shared<logging::CoutSink>());
  return RUN_ALL_TESTS();
}
