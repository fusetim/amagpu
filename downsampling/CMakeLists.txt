cmake_minimum_required(VERSION 3.0)
project(downsampling)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
  lib/
  common/
)

link_directories(
	${PROJECT_SOURCE_DIR}/OpenCL-ICD-Loader/inc
	${PROJECT_SOURCE_DIR}/OpenCL-ICD-Loader/inc/CL
)

IF (WIN32)
  SET(CL_LIBRARIES OpenCL)
ELSE (WIN32)
  SET(CL_LIBRARIES OpenCL)
ENDIF(WIN32)

add_definitions(-DSRC_PATH=\"${CMAKE_SOURCE_DIR}/\")
add_executable(downsampling downsampling.cpp downsampling.cl imageadp.cpp imageadp.cpp common/clutils.cpp lib/lodepng.cpp)
if(APPLE)
	target_link_libraries(downsampling "-framework OpenCL" )
else()
	target_link_libraries(downsampling ${CL_LIBRARIES})
endif()

add_subdirectory(OpenCL-ICD-Loader)
