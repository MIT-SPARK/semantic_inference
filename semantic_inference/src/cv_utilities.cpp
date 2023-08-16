#include "semantic_inference/cv_utilities.h"

#include <opencv2/imgproc/hal/interface.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/hal/hal.hpp>

namespace semantic_inference::helpers {

void resize(const cv::Mat& src, cv::Mat& dest, cv::Size dsize, int interpolation) {
  cv::Size ssize = src.size();
  double inv_scale_x = static_cast<double>(dsize.width) / ssize.width;
  double inv_scale_y = static_cast<double>(dsize.height) / ssize.height;

  cv::Mat temp;
  temp.create(dsize, src.type());

  cv::hal::resize(src.type(),
                  src.data,
                  src.step,
                  src.cols,
                  src.rows,
                  temp.data,
                  temp.step,
                  temp.cols,
                  temp.rows,
                  inv_scale_x,
                  inv_scale_y,
                  interpolation);

  // avoid aliasing issues when the input and output are the same
  dest = temp;
}

}  // namespace semantic_inference::helpers
