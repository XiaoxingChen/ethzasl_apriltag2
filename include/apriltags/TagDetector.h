#ifndef TAGDETECTOR_H
#define TAGDETECTOR_H

#include <vector>

#include "opencv2/opencv.hpp"

#include "apriltags//TagDetection.h"
#include "apriltags//TagFamily.h"
#include "apriltags//FloatImage.h"
#include "apriltags/Tag36h11.h"
namespace AprilTags {

class TagDetector {
public:
	
	const TagFamily thisTagFamily;

	//! Constructor
  // note: TagFamily is instantiated here from TagCodes
	TagDetector(const TagCodes& tagCodes, const size_t blackBorder=2) : thisTagFamily(tagCodes, blackBorder) {}

	TagDetector() : thisTagFamily(AprilTags::tagCodes36h11, 2) {}
	
	std::vector<TagDetection> extractTags(const cv::Mat& image);
	std::vector<TagDetection> extractTags(const FloatImage& image);
	std::vector<TagDetection> extractTags(const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>& image);
};

} // namespace
#endif
