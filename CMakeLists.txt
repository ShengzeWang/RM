cmake_minimum_required(VERSION 3.16)
project(rm_model LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

file(GLOB RM_MODEL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rm_model/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rm_model/models/*.cpp
)

add_library(rm_model STATIC ${RM_MODEL_SOURCES})
target_include_directories(rm_model PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(rm_model PUBLIC Threads::Threads)
target_compile_options(rm_model PRIVATE -Wall -Wextra -Wpedantic)

add_executable(rm_model_learner
  tools/learner/main.cpp
  tools/learner/load.cpp
)

target_include_directories(rm_model_learner PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/learner
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rm_model_learner PRIVATE rm_model)
target_compile_options(rm_model_learner PRIVATE -Wall -Wextra -Wpedantic)

add_executable(rm_model_inferencer
  tools/inferencer/main.cpp
  tools/learner/load.cpp
)

target_include_directories(rm_model_inferencer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/learner
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rm_model_inferencer PRIVATE rm_model)
target_compile_options(rm_model_inferencer PRIVATE -Wall -Wextra -Wpedantic)

if (UNIX AND NOT APPLE)
  target_link_libraries(rm_model_inferencer PRIVATE dl)
endif()
