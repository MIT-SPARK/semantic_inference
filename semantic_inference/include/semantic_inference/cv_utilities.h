#pragma once
#include <map>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

namespace semantic_inference::helpers {

void resize(const cv::Mat& src, cv::Mat& dst, cv::Size dsize, int interpolation);

}  // namespace semantic_inference::helpers
