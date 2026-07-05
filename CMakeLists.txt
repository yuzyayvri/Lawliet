cmake_minimum_required(VERSION 3.15)
project(CustomChessEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SFML 3 components MUST be capitalized exactly like this:
find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)

include_directories(src)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(chess_app ${SOURCES})

# Target names also require the SFML:: namespace prefix:
target_link_libraries(chess_app SFML::Graphics SFML::Window SFML::System)

add_custom_command(TARGET chess_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:chess_app>/assets
)
