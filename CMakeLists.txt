cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(super_pixel_inference CXX)

find_package(feature_matches_interface REQUIRED)

find_package(Torch REQUIRED)

add_library(${PROJECT_NAME} SHARED src/super_pixel_inference.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC feature_matches_interface)
target_link_libraries(${PROJECT_NAME} PRIVATE torch)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

target_link_options(${PROJECT_NAME} PRIVATE ${TORCH_CXX_FLAGS})

add_executable(superpoint_match src/match.cpp)
target_link_libraries(superpoint_match PUBLIC ${PROJECT_NAME})

include(GNUInstallDirs)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_directories(${PROJECT_NAME} PUBLIC
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}>
)

install(EXPORT ${PROJECT_NAME}
    DESTINATION share/${PROJECT_NAME}/cmake
    FILE ${PROJECT_NAME}Config.cmake
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h*"
)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS superpoint_match RUNTIME DESTINATION ${BINDIR})
