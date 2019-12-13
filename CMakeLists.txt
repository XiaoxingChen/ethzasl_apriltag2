cmake_minimum_required(VERSION 2.8)

project(ethzasl_apriltag2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
set(PYBIND11_PYTHON_VERSION 3.5)
find_package(pybind11 REQUIRED)

add_definitions(-fPIC -O3)
include_directories(include  ${EIGEN3_INCLUDE_DIRS})

#library
file(GLOB SOURCE_FILES "src/*.cc")
add_library(${PROJECT_NAME}  ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS}  pybind11::module)

add_library(py_ethztag MODULE ${SOURCE_FILES})
target_link_libraries(py_ethztag PRIVATE pybind11::module ${OpenCV_LIBS})
set_target_properties(py_ethztag PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

#demo
# if(NOT APPLE)
#   add_executable(apriltags_demo src/example/apriltags_demo.cpp src/example/Serial.cpp)
#   target_link_libraries(apriltags_demo ${PROJECT_NAME} v4l2)

#   add_executable(image_demo src/example/image_demo.cpp)
#   target_link_libraries(image_demo ${PROJECT_NAME})
# endif()


