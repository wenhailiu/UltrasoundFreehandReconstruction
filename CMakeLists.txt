cmake_minimum_required(VERSION 3.8)
project(freehand3d_US_reconstruction)

enable_language(CUDA)

#enable c++11
set(CMAKE_CXX_STANDARD 17)

#fine CUDA:
find_package(CUDA QUIET REQUIRED)

include_directories(freehand3d_US_reconstruction ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

#settings for VTK
set(VTK_DIR "/home/ultrast-s1/opensources/PlusBuild_build/Deps/vtk-bin")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
CUDA_INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS})

#settings for ITK
set(ITK_DIR "/home/ultrast-s1/opensources/PlusBuild_build/Deps/itk-bin")
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#settings for OPENCV
# set(OpenCV_DIR "/home/wenhai/Packages/openCV/opencv-4.1.0_build") #directory to cv build folder
find_package(OpenCV REQUIRED)

# if(CUDA_FOUND)
add_executable(
    FreehandReconstruction 
    freehand_reconstruction.cpp
    US_Freehand_Reconstruction.cpp
    freehand_reconstruction_GPU.cu
    )

#add_library( 
#    FreehandReconstruction SHARED 
#    freehand_reconstruction.cpp
#    US_Freehand_Reconstruction.cpp
#    freehand_reconstruction_GPU.cu
#    )

if ( TARGET Qt5::Core )
	get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
	string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
	set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()

target_link_libraries(
    FreehandReconstruction 
    ${CUDA_LIBRARIES}
    ${VTK_LIBRARIES} 
    ${ITK_LIBRARIES}
    ${OpenCV_LIBS}
    )



