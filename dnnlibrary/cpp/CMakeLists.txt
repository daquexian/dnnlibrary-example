# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.5.2)
project(daq_jni)

add_library(daq_jni
    ${PROJECT_SOURCE_DIR}/JavaWrapper.cpp)

add_library(daq
    STATIC
    IMPORTED)

set_target_properties(daq PROPERTIES IMPORTED_LOCATION
    ${PROJECT_SOURCE_DIR}/lib/libdaq.a)

target_include_directories(daq_jni
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(daq_jni
    daq)
