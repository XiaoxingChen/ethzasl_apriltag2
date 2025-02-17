#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include "apriltags/Tag36h11.h"
#include "apriltags/TagDetector.h"
#include "apriltags/TagDetection.h"
#include "pybind11/eigen.h"
#include <vector>

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
using namespace AprilTags;
using TagCornerMat = Eigen::Matrix<float, 4, 2, Eigen::RowMajor>;

TagCornerMat GetCorners(TagDetection& det)
{
  TagCornerMat corners;
  for(int i = 0; i < 4; i++)
  {
    corners(i, 0) = det.p[i].first;
    corners(i, 1) = det.p[i].second;
  }
  return corners;
}

PYBIND11_MODULE(py_ethztag,m)
{
  m.doc() = "pybind11 py_ethztag plugin";

  py::class_<AprilTags::TagDetection>(m, "TagDetection")
  .def(py::init())
  .def_readwrite("good", &TagDetection::good)
  .def("p", &GetCorners)
  .def_readonly("id", &TagDetection::id)
  .def("getRelativeTransform", &AprilTags::TagDetection::getRelativeTransform);

  py::class_<AprilTags::TagDetector>(m, "TagDetector")
  .def(py::init())
  .def("extractTags", (std::vector<TagDetection> (TagDetector::*)(const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>&) )&TagDetector::extractTags);
}