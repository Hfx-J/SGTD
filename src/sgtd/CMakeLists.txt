cmake_minimum_required(VERSION 3.10.0)
project(sgtd)

#SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++17 )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/include" "/opt/ros/noetic" "/home/beabbit/study_space/package/livox_ros/devel")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -std=c++0x -std=c++17 -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    math(EXPR PROC_NUM "${N} - 4")
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    # add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
include_directories(
  /home/beabbit/study_space/sgtd/Sophus
)
include_directories(
  /home/beabbit/anaconda3/lib/python3.9/site-packages/numpy/core/include
)
if (NOT OPENCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()
find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
find_package(yaml-cpp REQUIRED)
# find_package(Boost REQUIRED COMPONENTS serialization system filesystem regex
#                                        program_options iostreams thread)
# find_package(small_gicp REQUIRED)
if(BUILD_VGICP_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARY_DIRS})
endif()

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
  
)
# list(APPEND CMAKE_PREFIX_PATH "/usr/include")
# set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
find_package(Eigen3  REQUIRED)
message(Eigen: ${EIGEN3_INCLUDE_DIR})
# find_package(PCL REQUIRED)

# file(GLOB PCL_LIBRARIES /usr/lib/x86_64-linux-gnu/libpcl_*)
# set(PCL_INCLUDE_DIRS /usr/include/pcl-1.12)
find_package(PCL  REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(nlohmann_json REQUIRED)
message(PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS})
message(PCL_LIBRARIES: ${PCL_LIBRARIES})
find_package(Ceres REQUIRED)

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include
  )

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
if (OPENMP_FOUND)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
    endif ()
endif ()
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

### CUDA ###
if(BUILD_VGICP_CUDA)
  set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
  add_definitions(-DUSE_VGICP_CUDA)

  cuda_add_library(fast_vgicp_cuda SHARED
    src/fast_gicp/cuda/fast_vgicp_cuda.cu
    src/fast_gicp/cuda/brute_force_knn.cu
    src/fast_gicp/cuda/covariance_estimation.cu
    src/fast_gicp/cuda/covariance_estimation_rbf.cu
    src/fast_gicp/cuda/covariance_regularization.cu
    src/fast_gicp/cuda/gaussian_voxelmap.cu
    src/fast_gicp/cuda/find_voxel_correspondences.cu
    src/fast_gicp/cuda/compute_derivatives.cu
    src/fast_gicp/cuda/compute_mahalanobis.cu
    src/fast_gicp/cuda/ndt_cuda.cu
    src/fast_gicp/cuda/ndt_compute_derivatives.cu
  )
  # find_package(EIGEN REQUIRED)
  # find_package(NVBIO REQUIRED)
  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
    ${NVBIO_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
  )
  target_link_libraries(fast_vgicp_cuda
    ${catkin_LIBRARIES}
  )
  cuda_add_cublas_to_target(fast_vgicp_cuda)

  # add vgicp_cuda to libfast_gicp
  target_sources(fast_gicp PRIVATE
    src/fast_gicp/gicp/fast_vgicp_cuda.cpp
    src/fast_gicp/ndt/ndt_cuda.cpp
  )
  target_link_libraries(fast_gicp
    fast_vgicp_cuda
  )
  add_dependencies(fast_gicp fast_vgicp_cuda)
  if(catkin_FOUND)
    install(TARGETS fast_vgicp_cuda
      LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
  elseif (ament_cmake_FOUND)
    install(TARGETS fast_vgicp_cuda
      LIBRARY DESTINATION lib)
  endif()

  endif()

add_executable(create_semantic_graph src/get_json.cpp  src/ssc.cpp) 
target_link_libraries(create_semantic_graph ${catkin_LIBRARIES}  ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} yaml-cpp  nlohmann_json fast_gicp ${OpenCV_LIBS})
target_include_directories(create_semantic_graph PRIVATE ${PYTHON_INCLUDE_DIRS} )

# add_executable(create_semantic_graph_wild src/get_json_wild.cpp  src/ssc.cpp) 
# target_link_libraries(create_semantic_graph_wild ${catkin_LIBRARIES}  ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} yaml-cpp  nlohmann_json fast_gicp ${OpenCV_LIBS})
# target_include_directories(create_semantic_graph_wild PRIVATE ${PYTHON_INCLUDE_DIRS} )

add_executable(semantic_graph_localization src/semantic_graph_localization.cpp src/ssc.cpp src/STDesc.cpp ) 

target_link_libraries(semantic_graph_localization ${catkin_LIBRARIES}  ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} yaml-cpp  nlohmann_json fast_gicp ${OpenCV_LIBS} ${CERES_LIBRARIES} -lffi) 
target_include_directories(semantic_graph_localization PRIVATE ${PYTHON_INCLUDE_DIRS} include )

add_executable(local_map src/local_map.cpp  ) 
target_link_libraries(local_map ${catkin_LIBRARIES}  ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} yaml-cpp  nlohmann_json fast_gicp) 
target_include_directories(local_map PRIVATE ${PYTHON_INCLUDE_DIRS} )